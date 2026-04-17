from __future__ import annotations

import argparse
import json

from prepare import (
    compare_candidate,
    load_config,
    load_fixtures,
    require_memory_limit,
    split_fixtures,
)


def generate_text(model, tokenizer, prompt_tokens, *, max_tokens: int):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    def prompt_cache_for_model():
        prompt_cache = getattr(model, "_autoresearch_prompt_cache", None)
        if prompt_cache is None:
            prompt_cache = make_prompt_cache(model)
            model._autoresearch_prompt_cache = prompt_cache
            return prompt_cache

        for cache in prompt_cache:
            if hasattr(cache, "offset"):
                cache.offset = 0
            if hasattr(cache, "_idx"):
                cache._idx = 0
        return prompt_cache

    def sample_next(input_tokens):
        logits = model(input_tokens[None], cache=prompt_cache)
        return mx.argmax(logits[:, -1, :], axis=-1)

    prompt = mx.array(prompt_tokens)
    if prompt.size == 0:
        raise ValueError("Prompt must contain at least one token")

    prompt_cache = prompt_cache_for_model()
    if prompt.size > 1:
        model(prompt[:-1][None], cache=prompt_cache)
        mx.eval([cache.state for cache in prompt_cache])
        prompt = prompt[-1:]

    current = prompt
    eos_token_ids = set(tokenizer.eos_token_ids)
    generated_token_ids: list[int] = []
    next_token = sample_next(current)
    mx.async_eval(next_token)

    for token_count in range(max_tokens):
        if token_count + 1 < max_tokens:
            following_token = sample_next(next_token)
            mx.async_eval(following_token)

        mx.eval(next_token)

        token = int(next_token.item())
        if token in eos_token_ids:
            break

        generated_token_ids.append(token)
        if token_count + 1 < max_tokens:
            next_token = following_token

        if token_count % 256 == 0:
            mx.clear_cache()

    return {
        "token_ids": generated_token_ids,
        "text": "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the current generate.py candidate"
    )
    parser.add_argument("--full", action="store_true", help="Use the full fixture set")
    parser.add_argument(
        "--description",
        default="manual run",
        help="Short description of the candidate change",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config()
    fixtures = load_fixtures()
    quick_fixtures, full_fixtures = split_fixtures(config, fixtures)
    require_memory_limit(config)

    mode = "full" if args.full else "quick"
    selected_fixtures = full_fixtures if args.full else quick_fixtures
    result = compare_candidate(config, mode, selected_fixtures, args.description)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
