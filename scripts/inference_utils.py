"""Dependency-light helpers shared by model inference and unit tests."""


def unpack_generation_inputs(prepared_inputs):
    """Normalize tensor and BatchEncoding chat-template outputs."""
    if hasattr(prepared_inputs, "keys") and "input_ids" in prepared_inputs:
        model_inputs = {
            key: prepared_inputs[key] for key in prepared_inputs.keys()
        }
    else:
        model_inputs = {"input_ids": prepared_inputs}
    return model_inputs, model_inputs["input_ids"].shape[1]


def decode_generated_continuations(generated, input_length, tokenizer):
    """Decode only tokens generated after the padded prompt width."""
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    new_tokens = sequences[:, input_length:]
    return [text.strip().lower().rstrip(".!?,") for text in
            tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
