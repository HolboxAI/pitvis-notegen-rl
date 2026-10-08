"""Batch generation for evaluation and inference: vLLM when available, transformers otherwise."""
from __future__ import annotations

import os
from pathlib import Path

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


class Generator:
    def __init__(self, base: str, adapter: str | None, llm_cfg: dict, temperature: float = 0.0,
                 backend: str = "auto"):
        if not base:
            raise SystemExit("config.yaml: set llm.base")
        self.base, self.adapter, self.cfg, self.temperature = base, adapter, llm_cfg, temperature
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
        self.llm = LLM(model=self.base, enable_lora=bool(self.adapter), max_lora_rank=128,
                       max_model_len=self.cfg["max_model_len"], trust_remote_code=True,
                       gpu_memory_utilization=self.cfg["gpu_memory_utilization"],
                       tensor_parallel_size=self.cfg["tensor_parallel_size"], **extra)
        stop = self.cfg.get("stop") or []
        self.sp = SamplingParams(temperature=self.temperature, max_tokens=self.cfg["max_new_tokens"],
                                 stop=stop or None, include_stop_str_in_output=bool(stop))
        self.lora = None
        if self.adapter:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("note_adapter", 1, self.adapter)

    def _gen_vllm(self, convs):
        kw = dict(sampling_params=self.sp, lora_request=self.lora)
        try:
            outs = self.llm.chat(convs, chat_template_kwargs={"enable_thinking": False}, **kw)
        except TypeError:
            outs = self.llm.chat(convs, **kw)
        return [o.outputs[0].text for o in outs]

    # -- transformers -------------------------------------------------------
    def _init_hf(self):
        import torch
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.base, trust_remote_code=True)
        model = None
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
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

    def _gen_hf(self, convs):
        import torch
        outs = []
        for conv in convs:
            try:
                prompt = self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
            except TypeError:
                prompt = self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**ids, max_new_tokens=self.cfg["max_new_tokens"],
                                          do_sample=self.temperature > 0,
                                          temperature=self.temperature if self.temperature > 0 else None)
            text = self.tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            for s in self.cfg.get("stop") or []:
                if s in text:
                    text = text[:text.index(s) + len(s)]
            outs.append(text)
        return outs

    def generate(self, convs: list) -> list:
        return self._gen_vllm(convs) if self.backend == "vllm" else self._gen_hf(convs)
