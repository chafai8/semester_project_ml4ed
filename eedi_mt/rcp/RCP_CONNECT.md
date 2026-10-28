# Connecting to the EPFL RCP cluster from a terminal (runai + kubectl, Windows)

EPFL's RCP offers GPUs as containers through **Run:ai** (on Kubernetes). You never "log into a GPU node": you submit
a job (a container image + a command) that asks for N GPUs, and the scheduler places it on a node.
`runai` submits and monitors jobs; `kubectl` is used for what `runai` cannot do: listing the storage volumes and
copying files into and out of them.

> Sources: the public lab guides [epfml/getting-started](https://github.com/epfml/getting-started) and
> [epflneuroailab/rcp-access](https://github.com/epflneuroailab/rcp-access); the rest was read from this cluster
> with `rcp\probe_cluster.ps1` and `kubectl`.

## What we found on your account
| Item | Value |
|---|---|
| Run:ai project / namespace | `ml4ed-chafai` / `runai-ml4ed-chafai` (department `ml4ed`) |
| Guaranteed GPUs | **3** (`DESERVED GPUs`). Training jobs have no time limit; interactive jobs: 12 h, and stopped after 2 h with an idle GPU |
| Storage volumes (PVC) | `ml4ed-scratch` (use this one), `home` - both 100 Ti, ReadWriteMany. You may not create new volumes |
| CLI | the cluster accepts the **v1 syntax**: `runai list project`, `runai config project`, `runai submit`, `runai list jobs` |
| GPU pools | default = **A100 80GB** (`gpu001`-`gpu029`, 8 per node); own pools: `a100-40g`, `v100`, `h100`, `h200`, `b300` |
| Drivers | NVIDIA 580 (CUDA 13.0 runtime), Ubuntu 22.04 nodes |

## Setup (once)
- Be on the **EPFL network or VPN**. Use **PowerShell**.
- `runai.exe` and `kubectl.exe` must be in a folder on your PATH (classic `cmd` finds programs in the current folder
  even when they are not on PATH, which hides this problem).
- `runai login` (prints a URL + code, sign in with GASPAR), then `runai config project ml4ed-chafai`.
- Hide the deprecation message with `$env:SUPPRESS_DEPRECATION_MESSAGE="true"` (the helper scripts do this themselves).
- `kubectl config get-contexts` should show `rcp-caas-prod` with namespace `runai-ml4ed-chafai`.

## How our jobs use the cluster
- **Image:** `vllm/vllm-openai:v0.31.0-cu129` (vLLM, PyTorch and transformers preinstalled), overriding its default
  command with our own script (`rcp/run_step.sh`).
- **Storage:** `ml4ed-scratch` is the lab's shared scratch (10 TB), mounted at `/scratch`; everything of ours is in
  `/scratch/chafai/eedi_mt` (`code/`, `hf_cache/`, `cache/`, `logs/`, `.hf_token`). Anything outside the volume
  disappears when a job ends. Delete `hf_cache` when the project is done.
- **Files in and out:** `rcp\sync_scratch.ps1` starts a tiny CPU-only helper job with the volume mounted, copies with
  `kubectl cp`, and deletes the helper again.
- **Identity:** the volume root is owned by uid 222948 / gid 11159 with mode 2770, so nobody else can enter it (not even
  `root`, which the file server maps to a nobody user). Jobs therefore run as exactly those IDs with `--run-as-uid` and
  `--run-as-gid`. `--run-as-user` does not work from the Windows CLI (it prints "tagging with user and group id is not
  available" and the job gets an ID without access).
- **No wasted GPU time:** model downloads (up to ~175 GB) run in a CPU-only job; each GPU job checks the GPU type and memory
  first and stops at once on the wrong hardware; jobs have no automatic retry; a job frees its GPUs when its script ends.
- **GPU plan inside your quota of 3:** repair candidates 1 GPU + Qwen post-edit 2 GPUs together; Qwen3-32B judge 2 GPUs +
  GPT-OSS-120B judge 1 GPU together (set `$GptossTp = 2` in `config.ps1` and run them one after the other if gpt-oss
  does not fit on one GPU). All on A100 80GB.

## Everyday commands
```powershell
runai list jobs -p ml4ed-chafai              # what is running / finished
runai logs <job> -p ml4ed-chafai             # output of a job
runai describe job <job> -p ml4ed-chafai     # why is it Pending / Failed?
runai delete job <job> -p ml4ed-chafai       # stop / remove a job (stopping frees its GPUs)
kubectl get pods -n runai-ml4ed-chafai       # the pods behind the jobs
```
If a job stays `Pending`, `runai describe job` says why (no free GPU in the pool, image still pulling, ...).
Delete jobs you no longer need: a running job keeps its GPUs reserved.

## The run order is in README.md ("Order of operations").
