"""EXP-110b memory probe (D-152b): one forward and backward pass at a given sequence length, with the model built exactly as
exp088_sft_lora.py's train() builds it (bf16 QAT E2B, per-layer embeddings on the CPU, no vision or audio encoders, query-chunked
SDPA, rank-16 LoRA on gate, up, down and o_proj, gradient checkpointing, train mode). Prints one line:
PROBE <length> OK <peak GB> chunked_calls <n>   or   PROBE <length> OOM
Run as a separate process per length, so a failure cannot leave memory held in the training process.
Usage: python exp110b_memprobe.py <length>"""
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import torch  # noqa: E402

N = int(sys.argv[1])
src = open(os.path.join(HERE, "exp088_sft_lora.py"), encoding="utf-8").read()
E88 = {"__file__": os.path.join(HERE, "exp088_sft_lora.py"), "__name__": "exp088lib"}
exec(compile(src.split('with Experiment("EXP-088"')[0], "exp088", "exec"), E88)
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import Gemma4ForConditionalGeneration  # noqa: E402
from sne.chunked_attention import install  # noqa: E402
from sne.model import _CPUEmbedding  # noqa: E402

calls = {"n": 0}
fn = install()


def counted(*a, **k):
    if a[1].shape[-1] > 256 and a[1].shape[2] > 4096:
        calls["n"] += 1
    return fn(*a, **k)


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
ALL_ATTENTION_FUNCTIONS["sdpa"] = counted
path = snapshot_download(E88["QAT"], local_dir="/content/bf16_src")
model = Gemma4ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
for nm in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
    if getattr(model.model, nm, None) is not None:
        setattr(model.model, nm, None)
tm = model.model.language_model
ple = tm.embed_tokens_per_layer
tm.embed_tokens_per_layer = torch.nn.Identity()
model = model.cuda()
tm.embed_tokens_per_layer = _CPUEmbedding(ple, "cuda")
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)
for name, mod in list(model.named_modules()):
    if re.fullmatch(E88["TARGET"], name) and isinstance(mod, torch.nn.Linear):
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], E88["LoRALinear"](mod, E88["R"], E88["ALPHA"]))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.train()
torch.cuda.reset_peak_memory_stats()
ids = torch.randint(1000, 200000, (1, N), device="cuda")
keep = torch.zeros(N - 1, dtype=torch.bool, device="cuda")
keep[int(N * 0.55):] = True  # target share like the data (median 5,958 of 13,174 tokens)
try:
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1][keep]
    tgt = ids[0, 1:][keep]
    tot = torch.zeros((), device="cuda")
    cap = model.config.get_text_config().final_logit_softcapping

    def chunk_loss(hc, tc):  # as exp088's loss_of: logits recomputed in the backward pass (D-152c)
        lg = model.lm_head(hc).float()
        if cap:
            lg = torch.tanh(lg / cap) * cap
        return torch.nn.functional.cross_entropy(lg, tc, reduction="sum")

    import torch.utils.checkpoint
    for k in range(0, h.shape[0], 512):
        tot = tot + torch.utils.checkpoint.checkpoint(chunk_loss, h[k:k + 512], tgt[k:k + 512], use_reentrant=False)
    (tot / keep.sum()).backward()
    print(f"PROBE {N} OK {torch.cuda.max_memory_allocated() / 1e9:.2f} chunked_calls {calls['n']}", flush=True)
except torch.cuda.OutOfMemoryError:
    print(f"PROBE {N} OOM chunked_calls {calls['n']} peak_before_failure {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
