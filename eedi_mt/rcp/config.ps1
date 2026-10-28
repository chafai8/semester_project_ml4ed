# Shared settings for the RCP helper scripts (dot-sourced by submit_job.ps1 and sync_scratch.ps1).
# Edit here if your project / volume / image changes. Do not run this file directly.

$Project     = "ml4ed-chafai"            # runai list project
$Namespace   = "runai-ml4ed-chafai"      # MANAGED NAMESPACE column of the same table
$Pvc         = "ml4ed-scratch"           # kubectl get pvc -n <namespace>   (ReadWriteMany, so several jobs can mount it)
$Mount       = "/scratch"                # where the volume appears inside the containers
$Base        = "/scratch/chafai/eedi_mt" # everything of ours lives here: code/, hf_cache/, cache/, logs/, .hf_token
# The volume root is owned by uid 222948 / gid 11159 with mode 2770 (found with `stat /scratch` in a test pod), so only that
# user or group can enter it. The Windows runai cannot fill in your IDs for --run-as-user, so the jobs get them explicitly.
$RunAsUid    = 222948
$RunAsGid    = 11159
$Image       = "vllm/vllm-openai:v0.31.0-cu129"   # GPU jobs (vLLM + torch + transformers preinstalled)
# gpt-oss-120b (MXFP4, ~63 GB of weights) is meant to fit one 80 GB A100. If check_gptoss runs out of memory, set this to 2
# (then run the two judges one after the other: 2 + 2 GPUs would exceed the quota of 3).
$GptossTp    = 1
# Only if check_gptoss fails inside the attention kernels on the A100, try "TRITON_ATTN" here (vLLM attention backend name).
$VllmAttn    = ""
$HelperImage = "ubuntu:22.04"            # CPU-only helper job used to copy files to/from the volume
$HelperJob   = "eedi-sync"

$env:SUPPRESS_DEPRECATION_MESSAGE = "true"
$ProjectRoot = Split-Path -Parent $PSScriptRoot    # the eedi_mt folder (parent of rcp\)

function Require-Tool([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "'$Name' was not found on PATH (see rcp\RCP_CONNECT.md)"
    }
}

# Run a native command, show it, and stop on a non-zero exit code (unless -AllowFail).
function Run-Native {
    param([string]$Exe, [string[]]$CmdArgs, [switch]$AllowFail)
    Write-Host ("> " + $Exe + " " + ($CmdArgs -join " ")) -ForegroundColor DarkGray
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0 -and -not $AllowFail) {
        throw "$Exe failed with exit code $LASTEXITCODE"
    }
}
