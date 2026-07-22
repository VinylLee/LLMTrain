"""Dependency-light helpers shared by model inference and unit tests."""


def decode_generated_continuations(generated, input_length, tokenizer):
    """Decode only tokens generated after the padded prompt width."""
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    new_tokens = sequences[:, input_length:]
    return [text.strip().lower().rstrip(".!?,") for text in
            tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
