#!/usr/bin/env bash
# Runs INSIDE the cluster container. Started by submit_job.ps1 as:  bash run_step.sh <step>
# Steps (second run, "v2"): check_gpu | download_models | check_gptoss |
#   repair_candidates (1 GPU) | repair_postedit (2 GPU) | repair_collect (CPU) |
#   repair_judge_qwen (2 GPU) | repair_judge_gptoss (1 GPU) | repair_select (CPU) |
#   judge_qwen (2 GPU) | judge_gptoss (1 GPU)
set -uo pipefail

STEP="${1:?usage: run_step.sh <step>}"
BASE="${EEDI_BASE:-/scratch/chafai/eedi_mt}"    # set by submit_job.ps1 from rcp\config.ps1
CODE=$BASE/code
mkdir -p "$BASE/logs" "$BASE/hf_cache" "$BASE/cache"
LOG="$BASE/logs/${STEP}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "== step=$STEP  host=$(hostname)  date=$(date -Is)  uid=$(id -u) gid=$(id -g)"
echo "== log file: $LOG"

# ---- environment: all caches live on the scratch volume (home dirs are throwaway) ----------------------
[ -w "${HOME:-/nonexistent}" ] || export HOME=/tmp
export USER="${USER:-eedi}"                   # containers run as an arbitrary uid without a passwd entry; torch wants a user name
export LOGNAME="${LOGNAME:-$USER}"
export HF_HOME="$BASE/hf_cache"
export XDG_CACHE_HOME="$BASE/cache"
export VLLM_CACHE_ROOT="$BASE/cache/vllm"
export TRITON_CACHE_DIR="$BASE/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [ -f "$BASE/.hf_token" ]; then export HF_TOKEN="$(cat "$BASE/.hf_token")"; fi
PY="$(command -v python || command -v python3)"
GPTOSS_TP="${EEDI_GPTOSS_TP:-1}"                       # 1 GPU if gpt-oss-120b (MXFP4, ~63 GB) fits, else 2 (see config.ps1)
[ -n "${EEDI_VLLM_ATTN:-}" ] && export VLLM_ATTENTION_BACKEND="$EEDI_VLLM_ATTN"   # only if check_gptoss needs it
GPTOSS_ARGS=(--judge-name gptoss_120b --model openai/gpt-oss-120b --tp "$GPTOSS_TP" --dtype auto
             --chat-kwargs '{"reasoning_effort": "low"}' --max-tokens 1500 --keep-special)
QWEN_ARGS=(--judge-name qwen3_32b --model Qwen/Qwen3-32B --tp 2)

# The image's torchcodec (audio/video decoding, irrelevant for text) wants CUDA-13 libraries (libnvrtc.so.13) that this
# CUDA-12.9 image lacks, and fails with OSError, which vLLM's import guard does not catch (it catches ImportError only).
# A placeholder package that raises ImportError makes vLLM treat torchcodec as "not installed".
mkdir -p "$BASE/stubs/torchcodec"
cat > "$BASE/stubs/torchcodec/__init__.py" <<'STUBEOF'
raise ImportError("torchcodec is disabled in this project (text-only workload; the image's build needs CUDA 13 libraries)")
STUBEOF
export PYTHONPATH="$BASE/stubs${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE" || { echo "ERROR: $CODE missing - run sync_scratch.ps1 -Direction up first"; exit 2; }

# ---- helpers ------------------------------------------------------------------------------------------
# need_gpus <count> <min_MiB_each>: refuse to run on the wrong hardware instead of wasting a long run
need_gpus() {
  command -v nvidia-smi >/dev/null || { echo "ERROR: no nvidia-smi: this container has no GPU"; exit 3; }
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  local n; n=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | wc -l)
  [ "$n" -eq "$1" ] || { echo "ERROR: expected $1 GPU(s), container sees $n"; exit 3; }
  while read -r m; do
    [ "$m" -ge "$2" ] || { echo "ERROR: GPU has only ${m} MiB (< $2). Wrong GPU pool - stopping."; exit 3; }
  done < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
}

# gpu_env <ngpu>: hardware guard + models must already be in the cache (download_models did that)
gpu_env() {
  need_gpus "$1" 70000
  export HF_HUB_OFFLINE=1
}

run() {   # run <command...>: log it, run it, report the exit code
  echo "+ $*"
  "$@"
  local rc=$?
  echo "== finished step=$STEP exit_code=$rc date=$(date -Is)"
  exit $rc
}

# ---- steps --------------------------------------------------------------------------------------------
check_gpu() {
  need_gpus 1 70000
  touch "$BASE/.write_test" && rm "$BASE/.write_test" && echo "scratch volume is writable"
  "$PY" -m py_compile common.py translate_vllm.py judge_vllm.py repair_common.py repair_candidates.py repair_postedit.py repair_select.py && echo "scripts compile"
  export HF_HUB_OFFLINE=1      # like the real GPU steps: the models must already be in the cache
  "$PY" - <<'PYEOF'
import importlib.metadata as md, inspect
import torch
for p in ["vllm", "torch", "transformers", "huggingface_hub", "tokenizers", "numpy", "pillow"]:
    try:
        print(f"{p:16s}", md.version(p))
    except Exception:
        print(f"{p:16s} MISSING")
assert torch.cuda.is_available(), "CUDA not available"
d = torch.cuda.get_device_properties(0)
print("GPU:", d.name, round(d.total_memory / 2**30, 1), "GiB, compute capability", d.major, d.minor)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul OK:", float((x @ x).float().abs().mean()))
from vllm import LLM, SamplingParams
params = inspect.signature(LLM.chat).parameters
print("LLM.chat has chat_template_kwargs:", "chat_template_kwargs" in params)
assert "chat_template_kwargs" in params, "this vLLM version cannot disable Qwen3 thinking via chat_template_kwargs"
print("SamplingParams OK:", SamplingParams(temperature=0.0, max_tokens=8).max_tokens)
from transformers import AutoProcessor
print("transformers AutoProcessor import OK")
# the translation prompt must render offline from the cached model
proc = AutoProcessor.from_pretrained("google/translategemma-27b-it")
msgs = [{"role": "user", "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "fr", "text": "Hello [[1]] world"}]}]
print(repr(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))[:400])
print("CHECK_GPU PASSED")
PYEOF
  local rc=$?
  [ $rc -eq 0 ] || echo "CHECK_GPU FAILED (python exit code $rc)"
  echo "== finished step=$STEP exit_code=$rc date=$(date -Is)"
  exit $rc
}

download_models() {   # CPU-only job: downloading up to ~175 GB must not hold a GPU
  "$PY" - <<'PYEOF'
import glob, sys
from huggingface_hub import hf_hub_download, snapshot_download
repos = ["google/translategemma-27b-it", "Qwen/Qwen3-32B", "openai/gpt-oss-120b"]   # the first two are normally cached already
print("checking access (gated licenses) ...")
bad = []
for r in repos:
    try:
        hf_hub_download(r, "config.json")
        print("  access OK:", r)
    except Exception as e:
        print("  NO ACCESS:", r, "->", type(e).__name__, str(e)[:200])
        bad.append(r)
if bad:
    sys.exit("Accept the license on huggingface.co for %s and re-run sync_scratch.ps1 -Direction up -SetToken" % bad)
for r in repos:
    print("downloading", r, flush=True)
    p = snapshot_download(r, max_workers=8,
                          ignore_patterns=["*.pth", "original/*", "*.gguf", "*.onnx", "*.h5", "*.msgpack", "*.ot"])
    n = len(glob.glob(p + "/*.safetensors"))
    print("  done:", p, "| safetensors files:", n, flush=True)
    assert n > 0, "no .safetensors files found for " + r
print("ALL MODELS DOWNLOADED")
PYEOF
  local rc=$?
  echo "== finished step=$STEP exit_code=$rc date=$(date -Is)"
  exit $rc
}

check_gptoss() {   # does gpt-oss-120b load on this hardware, and does the judge parser read its answers?
  need_gpus "$GPTOSS_TP" 70000
  export HF_HUB_OFFLINE=1
  GPTOSS_TP="$GPTOSS_TP" "$PY" - <<'PYEOF'
import json, os, sys
sys.path.insert(0, ".")
from judge_vllm import PROMPT, parse
from vllm import LLM, SamplingParams
llm = LLM(model="openai/gpt-oss-120b", dtype="auto", tensor_parallel_size=int(os.environ["GPTOSS_TP"]), max_model_len=4096,
          gpu_memory_utilization=0.90, seed=0)
M = chr(92)   # a backslash, to write \( ... \) without escaping trouble
cases = [("Work out " + M + "( 3 " + M + "times 4 " + M + ")", "Calcule " + M + "( 3 " + M + "times 4 " + M + ")"),
         ("Work out " + M + "( 3 " + M + "times 4 " + M + ")", "Calcule " + M + "( 3 + 4 " + M + ")"),
         ("Which is the larger number?", "Quel nombre est le plus petit ?")]
convs = [[{"role": "user", "content": PROMPT.format(lang="French", field_desc="a maths question", src=s, tr=t)}] for s, t in cases]
sp = SamplingParams(temperature=0.0, max_tokens=1500, skip_special_tokens=False)
outs = llm.chat(convs, sp, chat_template_kwargs={"reasoning_effort": "low"})
ok = 0
for (s, t), o in zip(cases, outs):
    txt = o.outputs[0].text
    p = parse(txt)
    ok += p is not None
    print("-----", t); print("RAW (end):", repr(txt[-300:])); print("PARSED:", p, flush=True)
print("PARSED %d/%d" % (ok, len(cases)))
print("CHECK_GPTOSS PASSED" if ok == len(cases) else "CHECK_GPTOSS FAILED: see RAW above")
sys.stdout.flush(); os._exit(0 if ok == len(cases) else 1)
PYEOF
  local rc=$?
  echo "== finished step=$STEP exit_code=$rc date=$(date -Is)"
  exit $rc
}

case "$STEP" in
  check_gpu)         check_gpu ;;
  download_models)   download_models ;;

  check_gptoss)      check_gptoss ;;

  # ---- repair loop: new candidates for the flagged strings (needs work/repair/todo.jsonl, made by prep_repair.py) ----
  repair_candidates) gpu_env 1
                     run "$PY" repair_candidates.py ;;
  repair_postedit)   gpu_env 2
                     run "$PY" repair_postedit.py ;;
  repair_collect)    run "$PY" repair_collect.py ;;
  repair_judge_qwen) gpu_env 2
                     run "$PY" judge_vllm.py "${QWEN_ARGS[@]}" \
                         --translations-dir work/repair/cand_all --out-dir work/repair/judging ;;
  repair_judge_gptoss) gpu_env "$GPTOSS_TP"
                     run "$PY" judge_vllm.py "${GPTOSS_ARGS[@]}" \
                         --translations-dir work/repair/cand_all --out-dir work/repair/judging ;;
  repair_select)     run "$PY" repair_select.py ;;

  # ---- full judging of work/translations_v2 with the new judges (1-3 rubric) ----
  judge_qwen)        gpu_env 2
                     run "$PY" judge_vllm.py "${QWEN_ARGS[@]}" \
                         --translations-dir work/translations_v2 --out-dir work/judging_v2 ;;
  judge_gptoss)      gpu_env "$GPTOSS_TP"
                     run "$PY" judge_vllm.py "${GPTOSS_ARGS[@]}" \
                         --translations-dir work/translations_v2 --out-dir work/judging_v2 ;;

  *) echo "unknown step: $STEP"; exit 2 ;;
esac
