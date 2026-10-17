"""Vision-language model wrapper: loading, greedy extraction, and signal capture.

Heavy imports (``torch``, ``transformers``) are deferred so the rest of the package
stays importable without a GPU toolchain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from underwrite.config import Config
from underwrite.extract.prompts import SYSTEM_INSTRUCTION, build_extraction_prompt


@dataclass(frozen=True)
class TeacherForcedOutput:
    """Per-token signals over one generated response.

    ``logprobs``, ``entropies`` and ``margins`` each have one entry per response
    token, aligned with ``offsets`` into ``text``.
    """

    text: str
    n_prompt_tokens: int
    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    logprobs: tuple[float, ...]
    entropies: tuple[float, ...]
    margins: tuple[float, ...]
    hidden_states: dict[int, Any]
    offsets_aligned: bool


def load_vision_model(config: Config) -> tuple[Any, Any]:
    """Load the processor and the 4-bit quantized model described by ``config``."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as model_class
    except ImportError:  # pragma: no cover - older transformers releases
        from transformers import AutoModelForVision2Seq as model_class

    dtype = getattr(torch, config.model.compute_dtype)
    device_map = {"": 0} if torch.cuda.is_available() else "auto"

    model_kwargs: dict[str, Any] = {"device_map": device_map}
    if config.model.quantization == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    processor = AutoProcessor.from_pretrained(config.model.id, revision=config.model.revision)
    model = model_class.from_pretrained(
        config.model.id, revision=config.model.revision, **model_kwargs
    )
    model.eval()
    return processor, model


def build_messages(image: Any, fields: Sequence[str]) -> list[dict[str, Any]]:
    """Chat messages for a single extraction, with the image in the user turn."""
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_extraction_prompt(tuple(fields))},
            ],
        },
    ]


def _prompt_inputs(processor: Any, messages: Sequence[dict[str, Any]]) -> Any:
    return processor.apply_chat_template(
        list(messages),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def generate(
    processor: Any,
    model: Any,
    messages: Sequence[dict[str, Any]],
    max_new_tokens: int,
    do_sample: bool = False,
) -> str:
    """Greedy extraction: return only the newly generated text."""
    import torch

    inputs = _prompt_inputs(processor, messages).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
        )
    new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def pick_layers(hidden_state_count: int, fractions: Sequence[float]) -> list[int]:
    """Map depth fractions onto hidden-state indices.

    ``hidden_states[0]`` is the embedding output, so layer *L* ends at index
    ``L + 1``. Indices are clamped away from the embeddings and the final layer
    is included by construction when a fraction of 1.0 is requested.
    """
    last = hidden_state_count - 1
    chosen = {min(last, max(1, round(fraction * last))) for fraction in fractions}
    return sorted(chosen)


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    """First index at which ``needle`` occurs in ``haystack``, or ``None``."""
    if not needle or len(needle) > len(haystack):
        return None
    first = needle[0]
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start] == first and list(haystack[start : start + len(needle)]) == list(needle):
            return start
    return None


def teacher_forced(
    processor: Any,
    model: Any,
    messages: Sequence[dict[str, Any]],
    response_text: str,
    layers: Sequence[int] = (),
) -> TeacherForcedOutput:
    """Forward (image + prompt + response) once and collect per-token signals.

    The chat template may wrap the response in extra tokens, so the response's own
    tokens are located inside the templated sequence by subsequence search rather
    than assumed to begin at the end of the prompt.
    """
    import torch

    tokenizer = processor.tokenizer
    n_prompt = _prompt_inputs(processor, messages)["input_ids"].shape[-1]

    full_messages = list(messages) + [{"role": "assistant", "content": response_text}]
    full_inputs = processor.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    encoding = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = tuple(tuple(pair) for pair in encoding["offset_mapping"])
    text_ids = [int(value) for value in encoding["input_ids"]]

    response_ids = [int(value) for value in full_inputs["input_ids"][0, n_prompt:].tolist()]
    located = _find_subsequence(response_ids, text_ids)
    aligned = located is not None
    start = located if located is not None else 0
    window = response_ids[start : start + len(text_ids)]

    want_hidden = bool(layers)
    with torch.inference_mode():
        outputs = model(**full_inputs, output_hidden_states=want_hidden)

    # The distribution for a token lives at the position immediately before it.
    first = n_prompt - 1 + start
    region = outputs.logits[0, first : first + len(window)].float()
    log_probs = torch.log_softmax(region, dim=-1)

    realised = log_probs.gather(-1, torch.tensor(window, device=log_probs.device).unsqueeze(-1))
    realised = realised.squeeze(-1)
    probabilities = log_probs.exp()
    entropies = -(probabilities * log_probs).sum(-1)
    top_two = log_probs.topk(2, dim=-1).values
    margins = top_two[:, 0] - top_two[:, 1]

    hidden_states: dict[int, Any] = {}
    if want_hidden:
        for layer in layers:
            stacked = outputs.hidden_states[layer][0, n_prompt + start : n_prompt + start + len(window), :]
            hidden_states[layer] = stacked.to(torch.float16).cpu().numpy()

    return TeacherForcedOutput(
        text=response_text,
        n_prompt_tokens=int(n_prompt),
        token_ids=tuple(window),
        offsets=offsets,
        logprobs=tuple(float(value) for value in realised.tolist()),
        entropies=tuple(float(value) for value in entropies.tolist()),
        margins=tuple(float(value) for value in margins.tolist()),
        hidden_states=hidden_states,
        offsets_aligned=aligned,
    )
