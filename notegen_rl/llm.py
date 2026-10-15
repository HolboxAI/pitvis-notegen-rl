"""Batch generation for evaluation and inference: vLLM when available, transformers otherwise.

Conversations may hold images: a message's content is then a list of {"type": "text", "text"}
and {"type": "image", "path"} parts (see to_conversation), passed to vLLM as data URLs and to
transformers as PIL images."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from .prompts import IMAGE_TAG

# FlashInfer's top-k/top-p sampler JIT-compiles CUDA code at first use and needs nvcc (a full
# CUDA toolkit), which GPU images often lack. vLLM's PyTorch sampler is equivalent and needs none.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def list_checkpoints(root, resumable: bool = False) -> list:
    """`checkpoint-<step>` dirs under an ms-swift output dir (any vN-* run), sorted by step.
    resumable=True keeps only full checkpoints (optimizer + trainer state), usable for resume."""
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for p in root.rglob("checkpoint-*"):
        if not p.is_dir() or not (p / "adapter_config.json").exists():
            continue
        if resumable and not ((p / "trainer_state.json").exists() and (p / "optimizer.pt").exists()):
            continue
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        out.append((step, p.stat().st_mtime, p))
    return [str(p) for _, _, p in sorted(out)]


def find_latest_checkpoint(root, resumable: bool = False) -> str | None:
    """Checkpoint with the highest training step (resumed runs continue the step count)."""
    cks = list_checkpoints(root, resumable)
    return cks[-1] if cks else None


def to_conversation(row: dict, text_only: bool = False) -> list:
    """Dataset row -> chat messages. Each IMAGE_TAG in a message is replaced, in order, by the
    next file of row["images"]. text_only uses the same prompt without keyframes."""
    if text_only and row.get("messages_text"):
        return json.loads(row["messages_text"])
    images = iter(row.get("images") or [])
    out = []
    for m in row["messages"]:
        c = m["content"]
        if text_only or not isinstance(c, str) or IMAGE_TAG not in c:
            out.append(m)
            continue
        chunks, parts = c.split(IMAGE_TAG), []
        for k, chunk in enumerate(chunks):
            if chunk:
                parts.append({"type": "text", "text": chunk})
            if k < len(chunks) - 1:
                parts.append({"type": "image", "path": next(images)})
        out.append({"role": m["role"], "content": parts})
    return out


def _has_images(conv) -> bool:
    return any(isinstance(m["content"], list) for m in conv)


def _data_url(path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _for_vllm(conv) -> list:
    return [m if not isinstance(m["content"], list) else
            {"role": m["role"], "content": [p if p["type"] != "image" else
                                            {"type": "image_url", "image_url": {"url": _data_url(p["path"])}}
                                            for p in m["content"]]}
            for m in conv]


class Generator:
    def __init__(self, base: str, adapter: str | None, llm_cfg: dict, temperature: float = 0.0,
                 backend: str = "auto", max_images: int = 0):
        """max_images: most images any prompt carries (0 = text-only prompts)."""
        if not base:
            raise SystemExit("config.yaml: set llm.base")
        self.base, self.adapter, self.cfg, self.temperature = base, adapter, llm_cfg, temperature
        self.max_images = int(max_images)
        self.backend = backend
        if backend == "auto":
            try:
                import vllm  # noqa: F401
                self.backend = "vllm"
            except ImportError:
                self.backend = "transformers"
        self._init_vllm() if self.backend == "vllm" else self._init_hf()

    # -- vLLM ---------------------------------------------------------------
    def _init_vllm(self):
        from vllm import LLM, SamplingParams
        extra = {"quantization": self.cfg["quantization"]} if self.cfg.get("quantization") else {}
        if self.max_images:
            extra["limit_mm_per_prompt"] = {"image": self.max_images}
        self.llm = LLM(model=self.base, enable_lora=bool(self.adapter), max_lora_rank=128,
                       max_model_len=self.cfg["max_model_len"], trust_remote_code=True,
                       gpu_memory_utilization=self.cfg["gpu_memory_utilization"],
                       tensor_parallel_size=self.cfg["tensor_parallel_size"], **extra)
        stop = self.cfg.get("stop") or []
        extra_sp = {}
        banned = self._banned_ids(self.llm.get_tokenizer())
        if banned:  # e.g. MedGemma's reasoning-block opener: never start a reasoning block
            extra_sp["logit_bias"] = {i: -100.0 for i in banned}
        self.sp = SamplingParams(temperature=self.temperature, max_tokens=self.cfg["max_new_tokens"],
                                 stop=stop or None, include_stop_str_in_output=bool(stop),
                                 repetition_penalty=float(self.cfg.get("repetition_penalty") or 1.0),
                                 skip_special_tokens=False, **extra_sp)
        self.lora = None
        if self.adapter:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("note_adapter", 1, self.adapter)

    def _gen_vllm(self, convs):
        convs = [_for_vllm(c) for c in convs]
        kw = dict(sampling_params=self.sp, lora_request=self.lora)
        if self.cfg.get("response_prefix"):
            kw.update(add_generation_prompt=False, continue_final_message=True)
        try:
            outs = self.llm.chat(convs, chat_template_kwargs={"enable_thinking": False}, **kw)
        except TypeError:
            outs = self.llm.chat(convs, **kw)
        return [o.outputs[0].text for o in outs]

    # -- transformers -------------------------------------------------------
    def _init_hf(self):
        import torch
        from transformers import AutoProcessor, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.base, trust_remote_code=True)
        self.proc = AutoProcessor.from_pretrained(self.base, trust_remote_code=True) if self.max_images else None
        model = None
        classes = ("AutoModelForCausalLM", "AutoModelForImageTextToText")
        for cls_name in (classes[::-1] if self.max_images else classes):
            try:
                import transformers
                model = getattr(transformers, cls_name).from_pretrained(
                    self.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
                break
            except (ValueError, KeyError, AttributeError):
                continue
        if model is None:
            raise RuntimeError(f"could not load {self.base} with transformers")
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter).merge_and_unload()
        self.model = model.eval()

    def _banned_ids(self, tok) -> list:
        """Token ids of llm.suppress_tokens (tokens the model must never generate)."""
        ids = []
        for t in self.cfg.get("suppress_tokens") or []:
            i = tok.convert_tokens_to_ids(t)
            if i is not None and i != getattr(tok, "unk_token_id", None):
                ids.append(int(i))
        return ids

    def _turn_kwargs(self) -> dict:
        if self.cfg.get("response_prefix"):
            return {"add_generation_prompt": False, "continue_final_message": True}
        return {"add_generation_prompt": True}

    def _gen_hf(self, convs):
        import torch
        outs = []
        for conv in convs:
            if _has_images(conv) and self.proc is not None:
                from PIL import Image
                conv = [m if not isinstance(m["content"], list) else
                        {"role": m["role"], "content": [p if p["type"] != "image" else
                                                        {"type": "image", "image": Image.open(p["path"]).convert("RGB")}
                                                        for p in m["content"]]} for m in conv]
                ids = self.proc.apply_chat_template(conv, tokenize=True, return_dict=True, return_tensors="pt",
                                                    **self._turn_kwargs()).to(self.model.device)
                if "pixel_values" in ids:
                    ids["pixel_values"] = ids["pixel_values"].to(self.model.dtype)
            else:
                try:
                    prompt = self.tok.apply_chat_template(conv, tokenize=False, enable_thinking=False,
                                                          **self._turn_kwargs())
                except TypeError:
                    prompt = self.tok.apply_chat_template(conv, tokenize=False, **self._turn_kwargs())
                ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                banned = self._banned_ids(self.tok)
                gen = self.model.generate(**ids, max_new_tokens=self.cfg["max_new_tokens"],
                                          repetition_penalty=float(self.cfg.get("repetition_penalty") or 1.0),
                                          **({"suppress_tokens": banned} if banned else {}),
                                          do_sample=self.temperature > 0,
                                          temperature=self.temperature if self.temperature > 0 else None)
            text = self.tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
            text = text.split(self.tok.eos_token)[0] if self.tok.eos_token else text
            text = text.replace("<end_of_turn>", "")
            for s in self.cfg.get("stop") or []:
                if s in text:
                    text = text[:text.index(s) + len(s)]
            outs.append(text)
        return outs

    def _prefixed(self, convs: list) -> list:
        """With llm.response_prefix, each conversation ends in an assistant turn holding the prefix,
        which generation continues; the returned text is the continuation only."""
        prefix = self.cfg.get("response_prefix")
        return [c + [{"role": "assistant", "content": prefix}] for c in convs] if prefix else convs

    def generate(self, convs: list) -> list:
        convs = self._prefixed(convs)
        return self._gen_vllm(convs) if self.backend == "vllm" else self._gen_hf(convs)
