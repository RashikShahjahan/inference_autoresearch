from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from mlx_lm.models import cache
from mlx_lm.models.cache import (
    KVCache,
    RotatingKVCache,
    TokenBuffer,
)

DEFAULT_MAX_TOKENS = 100

# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([p + [0] * (max_length - len(p)) for p in prompts])


@dataclass
class BatchStats:
    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


def _merge_caches(caches):
    batch_cache = []

    if not caches:
        return batch_cache

    for i in range(len(caches[0])):
        if hasattr(caches[0][i], "merge"):
            batch_cache.append(caches[0][i].merge([c[i] for c in caches]))
        else:
            raise ValueError(
                f"{type(caches[0][i])} does not yet support batching with history"
            )
    return batch_cache


def _extend_cache(cache_a, cache_b):
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a


def _build_trie(sequences):
    trie = {}
    for idx, seq in enumerate(sequences):
        node = trie
        try:
            for tok in seq:
                node = node.setdefault(tok, {})
            node["__match__"] = (tuple(seq), idx)
        except TypeError:
            node = node.setdefault(seq, {})
            node["__match__"] = ((seq,), idx)

    queue = deque()
    for key, child in trie.items():
        if key == "__match__":
            continue
        child["__fail__"] = trie
        queue.append(child)

    while queue:
        parent = queue.popleft()
        for key, child in parent.items():
            if key in ("__fail__", "__match__"):
                continue
            queue.append(child)
            fail = parent["__fail__"]
            while key not in fail and fail is not trie:
                fail = fail["__fail__"]
            child["__fail__"] = fail[key] if key in fail else trie
            if "__match__" not in child and "__match__" in child["__fail__"]:
                child["__match__"] = child["__fail__"]["__match__"]

    return trie


def _step_trie(node, trie, x):
    while x not in node and node is not trie:
        node = node["__fail__"]
    if x in node:
        node = node[x]
    return node


class SequenceStateMachine:
    def __init__(self, transitions=None, initial="normal"):
        transitions = transitions or {}
        self._initial = initial
        self._states = {}
        for src, edges in transitions.items():
            sequences, dst = zip(*edges)
            self._states[src] = (_build_trie(sequences), dst)
        if not self._states:
            self._states[initial] = (_build_trie([]), [])

    def __deepcopy__(self, memo):
        new = object.__new__(SequenceStateMachine)
        new._initial = self._initial
        new._states = self._states
        return new

    def make_state(self):
        return (self._initial, self._states[self._initial][0], self._states)

    @staticmethod
    def match(state, x):
        s, n, states = state
        n = _step_trie(n, states[s][0], x)

        seq = None
        match = n.get("__match__")
        if match is not None:
            seq = match[0]
            s = states[s][1][match[1]]
            n = states[s][0] if s is not None else None

        return (s, n, states), seq, s


class PromptProcessingBatch:
    @dataclass
    class Response:
        uid: int
        progress: tuple
        end_of_segment: bool
        end_of_prompt: bool

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        caches: List[List[Any]],
        tokens: Optional[List[List[int]]] = None,
        prefill_step_size: int = 2048,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        fallback_sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        state_machines: Optional[List[SequenceStateMachine]] = None,
        max_tokens: Optional[List[int]] = None,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = _merge_caches(caches)
        self.tokens = tokens if tokens is not None else [[] for _ in uids]

        self.prefill_step_size = prefill_step_size
        self.samplers = samplers if samplers is not None else []
        self.fallback_sampler = fallback_sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = (
            logits_processors if logits_processors is not None else []
        )
        self.state_machines = (
            state_machines
            if state_machines is not None
            else [SequenceStateMachine()] * len(uids)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else [DEFAULT_MAX_TOKENS] * len(self.uids)
        )

    def __len__(self):
        return len(self.uids)

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def extend(self, batch):
        if not any(self.samplers):
            self.samplers = [None] * len(self.uids)
        if not any(self.logits_processors):
            self.logits_processors = [None] * len(self.uids)

        samplers = batch.samplers if any(batch.samplers) else [None] * len(batch.uids)
        logits_processors = (
            batch.logits_processors
            if any(batch.logits_processors)
            else [None] * len(batch.uids)
        )

        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(samplers)
        self.logits_processors.extend(logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.state_machines.extend(batch.state_machines)

    def split(self, indices: List[int]):
        indices = sorted(indices)
        index_set = set(indices)
        indices_left = [idx for idx in range(len(self.uids)) if idx not in index_set]
        new_batch = self.__class__(
            model=self.model,
            uids=[self.uids[idx] for idx in indices],
            caches=[self.extract_cache(idx) for idx in indices],
            tokens=[self.tokens[idx] for idx in indices],
            prefill_step_size=self.prefill_step_size,
            samplers=[self.samplers[idx] for idx in indices],
            fallback_sampler=self.fallback_sampler,
            logits_processors=[self.logits_processors[idx] for idx in indices],
            state_machines=[self.state_machines[idx] for idx in indices],
            max_tokens=[self.max_tokens[idx] for idx in indices],
        )
        self.filter(indices_left)
        return new_batch

    def filter(self, keep: List[int]):
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        if any(self.samplers):
            self.samplers = [self.samplers[idx] for idx in keep]
        else:
            self.samplers = [None] * len(keep)
        if any(self.logits_processors):
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        else:
            self.logits_processors = [[]] * len(keep)
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.state_machines = [self.state_machines[idx] for idx in keep]

    def prompt(self, tokens: List[List[int]]):
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")

        if not tokens:
            return

        for sti, ti in zip(self.tokens, tokens):
            sti += ti

        lengths = [len(p) for p in tokens]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]
        max_padding = max(padding)

        if max_padding > 0:
            tokens = _right_pad_prompts(tokens, max_length=max_length)
            for c in self.prompt_cache:
                c.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)

        while tokens.shape[1] > 0:
            n_to_process = min(self.prefill_step_size, tokens.shape[1])
            self.model(tokens[:, :n_to_process], cache=self.prompt_cache)
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()
            tokens = tokens[:, n_to_process:]

        if max_padding > 0:
            for c in self.prompt_cache:
                c.finalize()
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()

    def generate(self, tokens: List[List[int]]):
        if any(len(t) > 1 for t in tokens):
            self.prompt([t[:-1] for t in tokens])

        last_token = mx.array([t[-1] for t in tokens])

        generation = GenerationBatch(
            self.model,
            self.uids,
            last_token,
            self.prompt_cache,
            self.tokens,
            self.samplers,
            self.fallback_sampler,
            self.logits_processors,
            self.state_machines,
            self.max_tokens,
        )

        self.uids = []
        self.prompt_cache = []
        self.tokens = []
        self.samplers = []
        self.logits_processors = []
        self.max_tokens = []

        return generation

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Callable[[mx.array], mx.array],
        prefill_step_size: int = 2048,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            prefill_step_size=prefill_step_size,
            uids=[],
            caches=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            state_machines=[],
        )


class GenerationBatch:
    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]
        current_state: Optional[str]
        match_sequence: Optional[List[int]]
        prompt_cache: Optional[List[Any]]
        all_tokens: Optional[List[int]]

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        tokens: List[List[int]],
        samplers: Optional[List[Callable[[mx.array], mx.array]]],
        fallback_sampler: Callable[[mx.array], mx.array],
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ],
        state_machines: List[SequenceStateMachine],
        max_tokens: List[int],
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.tokens = tokens

        self.samplers = samplers
        self.fallback_sampler = fallback_sampler
        self.logits_processors = logits_processors
        self.state_machines = state_machines
        self.max_tokens = max_tokens

        if self.samplers and len(self.samplers) != len(self.uids):
            raise ValueError("Insufficient number of samplers provided")
        if self.logits_processors and len(self.logits_processors) != len(self.uids):
            raise ValueError("Insufficient number of logits_processors provided")

        self._current_tokens = None
        self._current_logprobs = []
        self._next_tokens = inputs
        self._next_logprobs = []
        self._token_context = [TokenBuffer(t) for t in tokens]
        self._num_tokens = [0] * len(self.uids)
        self._matcher_states = [m.make_state() for m in state_machines]

        if self.uids:
            self._step()

    def __len__(self):
        return len(self.uids)

    def extend(self, batch):
        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(batch.samplers)
        self.logits_processors.extend(batch.logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.state_machines.extend(batch.state_machines)

        if self._current_tokens is None:
            self._current_tokens = batch._current_tokens
            self._current_logprobs = batch._current_logprobs
        elif batch._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, batch._current_tokens]
            )
            self._current_logprobs.extend(batch._current_logprobs)

        if self._next_tokens is None:
            self._next_tokens = batch._next_tokens
            self._next_logprobs = batch._next_logprobs
        elif batch._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, batch._next_tokens])
            self._next_logprobs.extend(batch._next_logprobs)

        self._token_context.extend(batch._token_context)
        self._num_tokens.extend(batch._num_tokens)
        self._matcher_states.extend(batch._matcher_states)

    def _step(self) -> Tuple[List[int], List[mx.array]]:
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens

        logits = self.model(inputs[:, None], cache=self.prompt_cache)
        logits = logits[:, -1, :]

        token_context = []
        if any(self.logits_processors):
            token_context = [
                tc.update_and_fetch(inputs[i : i + 1])
                for i, tc in enumerate(self._token_context)
            ]
            processed_logits = []
            for e in range(len(self.uids)):
                sample_logits = logits[e : e + 1]
                for processor in self.logits_processors[e]:
                    sample_logits = processor(token_context[e], sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        if any(self.samplers):
            all_samples = []
            for e in range(len(self.uids)):
                sample_sampler = self.samplers[e] or self.fallback_sampler
                sampled = sample_sampler(logprobs[e : e + 1])
                all_samples.append(sampled)
            sampled = mx.concatenate(all_samples, axis=0)
        else:
            sampled = self.fallback_sampler(logprobs)

        self._next_tokens = sampled
        self._next_logprobs = list(logprobs)
        mx.async_eval(self._next_tokens, self._next_logprobs, token_context)

        mx.eval(inputs, self._current_logprobs)
        inputs = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs):
            sti.append(ti)

        return inputs, self._current_logprobs

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def filter(self, keep: List[int]):
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        if any(self.samplers):
            self.samplers = [self.samplers[idx] for idx in keep]
        if any(self.logits_processors):
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.state_machines = [self.state_machines[idx] for idx in keep]

        self._next_tokens = self._next_tokens[keep] if keep else None
        self._next_logprobs = [self._next_logprobs[idx] for idx in keep]
        self._token_context = [self._token_context[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        self._matcher_states = [self._matcher_states[idx] for idx in keep]

    def next(self) -> List[Response]:
        if not self.uids:
            return []

        tokens, logprobs = self._step()

        keep = []
        responses = []
        for i in range(len(self.uids)):
            finish_reason = None
            match_sequence = None

            self._num_tokens[i] += 1
            if self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            self._matcher_states[i], match_sequence, current_state = (
                self.state_machines[i].match(self._matcher_states[i], tokens[i])
            )
            if match_sequence is not None and current_state is None:
                finish_reason = "stop"

            if finish_reason is not None:
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=finish_reason,
                        current_state=current_state,
                        match_sequence=match_sequence,
                        prompt_cache=self.extract_cache(i),
                        all_tokens=self.tokens[i],
                    )
                )
            else:
                keep.append(i)
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=None,
                        match_sequence=match_sequence,
                        current_state=current_state,
                        prompt_cache=None,
                        all_tokens=None,
                    )
                )

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Callable[[mx.array], mx.array],
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            uids=[],
            inputs=mx.array([], dtype=mx.uint32),
            prompt_cache=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            state_machines=[],
        )


class BatchGenerator:
    def __init__(
        self,
        model: nn.Module,
        max_tokens: int = 128,
        stop_tokens: Optional[Sequence[Sequence[int]]] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        max_kv_size: Optional[int] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = logits_processors or []
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.max_kv_size = max_kv_size

        self._default_state_machine = SequenceStateMachine(
            {"normal": [(seq, None) for seq in stop_tokens]} if stop_tokens else {},
            initial="normal",
        )
        self._uid_count = 0
        self._prompt_batch = PromptProcessingBatch.empty(
            self.model,
            self.sampler,
            prefill_step_size=prefill_step_size,
        )
        self._generation_batch = GenerationBatch.empty(self.model, self.sampler)
        self._unprocessed_sequences = deque()
        self._currently_processing = []

        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0

        if mx.metal.is_available():
            self._old_wired_limit = mx.set_wired_limit(
                mx.device_info()["max_recommended_working_set_size"]
            )
        else:
            self._old_wired_limit = None

    def close(self):
        if self._old_wired_limit is not None:
            mx.synchronize(generation_stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None

    def __del__(self):
        self.close()

    @contextlib.contextmanager
    def stats(self, stats=None):
        stats = stats or BatchStats()
        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        tic = time.perf_counter()
        try:
            yield stats
        finally:
            toc = time.perf_counter()
            total_time = toc - tic
            gen_time = total_time - self._prompt_time_counter
            stats.prompt_tokens += self._prompt_tokens_counter
            stats.prompt_time += self._prompt_time_counter
            stats.prompt_tps = stats.prompt_tokens / stats.prompt_time
            stats.generation_tokens += self._gen_tokens_counter
            stats.generation_time += gen_time
            stats.generation_tps = stats.generation_tokens / stats.generation_time
            stats.peak_memory = max(stats.peak_memory, mx.get_peak_memory() / 1e9)

    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        state_machines: Optional[List[SequenceStateMachine]] = None,
    ):
        return self.insert_segments(
            [[p] for p in prompts],
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            state_machines,
        )

    def insert_segments(
        self,
        segments: List[List[List[int]]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        state_machines: Optional[List[SequenceStateMachine]] = None,
    ):
        uids = []

        max_tokens = max_tokens or [self.max_tokens] * len(segments)
        all_tokens = all_tokens or [[] for _ in segments]
        samplers = samplers or [None] * len(segments)
        logits_processors = logits_processors or (
            [self.logits_processors] * len(segments)
        )
        state_machines = state_machines or (
            [self._default_state_machine] * len(segments)
        )

        caches = caches or [None] * len(segments)
        for i in range(len(segments)):
            if caches[i] is None:
                caches[i] = self._make_new_cache()

        for seq, m, c, at, s, lp, sm in zip(
            segments,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            state_machines,
        ):
            seq = list(seq)
            if len(seq[-1]) != 1:
                seq.append(seq[-1][-1:])
                seq[-2] = seq[-2][:-1]
            self._unprocessed_sequences.append(
                (self._uid_count, seq, m, c, at, s, lp, sm)
            )
            uids.append(self._uid_count)
            self._uid_count += 1

        return uids

    def _make_new_cache(self):
        if self.max_kv_size is None:
            return cache.make_prompt_cache(self.model)

        return [
            (
                RotatingKVCache(max_size=self.max_kv_size)
                if isinstance(ci, KVCache)
                else ci
            )
            for ci in cache.make_prompt_cache(self.model)
        ]

    def _make_batch(self, n: int):
        uids = []
        caches = []
        tokens = []
        samplers = []
        logits_processors = []
        max_tokens = []
        state_machines = []

        for _ in range(n):
            sequence = self._unprocessed_sequences.popleft()
            uids.append(sequence[0])
            caches.append(sequence[3])
            tokens.append(sequence[4])
            samplers.append(sequence[5])
            logits_processors.append(sequence[6])
            max_tokens.append(sequence[2])
            state_machines.append(sequence[7])
            self._currently_processing.append(
                [sequence[1], 0, sum(len(s) for s in sequence[1])]
            )

        return PromptProcessingBatch(
            model=self.model,
            uids=uids,
            caches=caches,
            tokens=tokens,
            prefill_step_size=self.prefill_step_size,
            samplers=samplers,
            fallback_sampler=self.sampler,
            logits_processors=logits_processors,
            state_machines=state_machines,
            max_tokens=max_tokens,
        )

    def _next(self):
        generation_responses = []
        prompt_responses = []

        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        n = min(
            self.prefill_batch_size - len(self._prompt_batch),
            self.completion_batch_size - len(self._generation_batch),
            len(self._unprocessed_sequences),
        )
        if n > 0:
            self._prompt_batch.extend(self._make_batch(n))

        keep = []
        split = []
        for i, seq in enumerate(self._currently_processing):
            segments = seq[0]
            if len(segments) == 1 and len(segments[0]) == 1:
                split.append(i)
            else:
                keep.append(i)

        if split:
            last_inputs = [self._currently_processing[i][0][0] for i in split]
            progress = [(self._currently_processing[i][2],) * 2 for i in split]
            self._currently_processing = [self._currently_processing[i] for i in keep]
            gen_batch = self._prompt_batch.split(split).generate(last_inputs)
            for i, p in enumerate(progress):
                prompt_responses.append(
                    PromptProcessingBatch.Response(
                        gen_batch.uids[i],
                        p,
                        True,
                        True,
                    )
                )
            self._generation_batch.extend(gen_batch)

        prompts = []
        for i, seq in enumerate(self._currently_processing):
            response = PromptProcessingBatch.Response(
                self._prompt_batch.uids[i], 0, False, False
            )
            segments = seq[0]
            n = min(len(segments[0]), self.prefill_step_size)
            prompts.append(segments[0][:n])
            segments[0] = segments[0][n:]
            if len(segments[0]) == 0:
                segments.pop(0)
                response.end_of_segment = True
            seq[1] += len(prompts[-1])
            response.progress = (seq[1], seq[2])
            prompt_responses.append(response)

        self._prompt_tokens_counter += sum(len(p) for p in prompts)
        tic = time.perf_counter()
        self._prompt_batch.prompt(prompts)
        toc = time.perf_counter()
        self._prompt_time_counter += toc - tic

        return prompt_responses, generation_responses

    def next_generated(self):
        with mx.stream(generation_stream):
            while True:
                prompt_responses, generation_responses = self._next()
                if not generation_responses and prompt_responses:
                    continue
                return generation_responses


@dataclass
class BatchResponse:
    token_ids: List[List[int]]
    texts: List[str]
    stats: BatchStats
    caches: Optional[List[List[Any]]]


def batch_generate(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[List[int]],
    prompt_caches: Optional[List[List[Any]]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    return_prompt_caches: bool = False,
    **kwargs,
) -> BatchResponse:
    """
    Generate responses for a batch of tokenized prompts.

    Args:
        model: The language model.
        tokenizer: Hugging Face tokenizer.
        prompts: List of token-id prompts.
        prompt_caches: Optional precomputed prompt caches, one per prompt.
        max_tokens: Max output tokens as int or per-prompt list.
        verbose: Whether to print progress and stats.
        return_prompt_caches: Whether to return final prompt caches.
        **kwargs: Passed through to BatchGenerator.

    Returns:
        BatchResponse with decoded texts, stats, and optional caches.
    """
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        **kwargs,
    )

    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    uids = gen.insert(prompts, max_tokens, caches=prompt_caches)
    results = {uid: [] for uid in uids}
    final_prompt_caches = {}

    with gen.stats() as stats:
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    if return_prompt_caches:
                        final_prompt_caches[r.uid] = r.prompt_cache
                    if verbose:
                        fin += 1
                        print(
                            f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                            end="\r",
                        )

                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)

    gen.close()

    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")

    token_ids = [results[uid] for uid in uids]
    texts = [tokenizer.decode(tokens) for tokens in token_ids]
    caches = (
        [final_prompt_caches[uid] for uid in uids] if return_prompt_caches else None
    )

    if verbose:
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, "
            f"{stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")

    return BatchResponse(token_ids, texts, stats, caches)


def generate_text(model, tokenizer, prompt_tokens_batch, *, max_tokens: int):
    response = batch_generate(
        model,
        tokenizer,
        prompt_tokens_batch,
        max_tokens=max_tokens,
    )
    return [
        {
            "token_ids": token_ids,
            "text": text,
        }
        for token_ids, text in zip(response.token_ids, response.texts)
    ]


def profile_batch_generate_metal(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[List[int]],
    *,
    max_tokens: Union[int, List[int]] = 128,
    trace_path: Union[str, Path] = "state/batch_generate_profile.gputrace",
    warmup: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Capture a single representative ``batch_generate`` call as a Metal trace."""
    if not mx.metal.is_available():
        raise RuntimeError("Metal profiling requires an Apple Silicon / Metal device")

    trace_path = Path(trace_path)
    if trace_path.suffix != ".gputrace":
        raise ValueError("trace_path must end with .gputrace")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    synchronize = getattr(mx, "synchronize", None)
    reset_peak_memory = getattr(mx, "reset_peak_memory", None)
    get_peak_memory = getattr(mx, "get_peak_memory", None)

    if warmup:
        batch_generate(
            model,
            tokenizer,
            prompts,
            max_tokens=max_tokens,
            **kwargs,
        )
        if callable(synchronize):
            synchronize()

    if callable(reset_peak_memory):
        reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()
    if callable(synchronize):
        synchronize()

    started = time.perf_counter()
    capture_started = False
    try:
        mx.metal.start_capture(str(trace_path))
        capture_started = True
        response = batch_generate(
            model,
            tokenizer,
            prompts,
            max_tokens=max_tokens,
            **kwargs,
        )
        if callable(synchronize):
            synchronize()
    finally:
        if capture_started:
            mx.metal.stop_capture()

    elapsed = time.perf_counter() - started
    peak_memory_bytes = (
        int(get_peak_memory())
        if callable(get_peak_memory)
        else int(mx.metal.get_peak_memory())
    )

    return {
        "trace_path": str(trace_path),
        "prompt_count": len(prompts),
        "output_tokens": sum(len(token_ids) for token_ids in response.token_ids),
        "elapsed_seconds": round(elapsed, 4),
        "peak_metal_mb": round(peak_memory_bytes / 1024 / 1024, 1),
        "warmup": warmup,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the current generate.py")
    parser.add_argument(
        "--description",
        default="manual run",
        help="Short description of the current experiment",
    )
    parser.add_argument(
        "--metal-profile-path",
        default=None,
        help="Write a Metal GPU trace for one representative batch_generate call",
    )
    parser.add_argument(
        "--metal-profile-fixture-index",
        type=int,
        default=0,
        help="Fixture index to use for representative Metal profiling",
    )
    return parser


def main() -> int:
    from prepare import (
        build_prompt,
        compare_candidate,
        load_config,
        load_fixtures,
        load_model_and_tokenizer,
        max_tokens_for_fixture,
        require_memory_limit,
    )

    parser = build_parser()
    args = parser.parse_args()

    config = load_config()
    fixtures = load_fixtures()
    require_memory_limit(config)

    if args.metal_profile_path is not None:
        fixture_index = args.metal_profile_fixture_index
        if fixture_index < 0 or fixture_index >= len(fixtures):
            raise ValueError(
                f"metal profile fixture index {fixture_index} is out of range for {len(fixtures)} fixtures"
            )

        model, tokenizer = load_model_and_tokenizer(config)
        fixture = fixtures[fixture_index]
        prompt = build_prompt(tokenizer, config, fixture.source_text)
        result = profile_batch_generate_metal(
            model,
            tokenizer,
            [prompt],
            max_tokens=max_tokens_for_fixture(config, fixture),
            trace_path=args.metal_profile_path,
        )
        print(json.dumps(result, indent=2))
        return 0

    result = compare_candidate(
        config,
        fixtures,
        args.description,
        generate_text,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"promoted", "incumbent"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
