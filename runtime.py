def generate_text(model, tokenizer, prompt_tokens, *, max_tokens: int):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    prompt = mx.array(prompt_tokens)
    if prompt.size == 0:
        raise ValueError("Prompt must contain at least one token")

    prompt_cache = make_prompt_cache(model)
    prefill_step_size = 2048

    while prompt.size > 1:
        n_to_process = min(prefill_step_size, prompt.size - 1)
        model(prompt[:n_to_process][None], cache=prompt_cache)
        mx.eval([cache.state for cache in prompt_cache])
        prompt = prompt[n_to_process:]
        mx.clear_cache()

    current = prompt
    eos_token_ids = set(tokenizer.eos_token_ids)
    generated_token_ids: list[int] = []

    for token_count in range(max_tokens):
        logits = model(current[None], cache=prompt_cache)
        logits = logits[:, -1, :]
        next_token = mx.argmax(logits, axis=-1)
        mx.eval(next_token)

        token = int(next_token.item())
        if token in eos_token_ids:
            break

        generated_token_ids.append(token)
        current = next_token

        if token_count % 256 == 0:
            mx.clear_cache()

    return {
        "token_ids": generated_token_ids,
        "text": tokenizer.decode(generated_token_ids).strip(),
    }
