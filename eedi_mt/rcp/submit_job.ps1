# Submits ONE step of the pipeline to the RCP cluster as a non-interactive ("training") job.
#
#   powershell -ExecutionPolicy Bypass -File .\rcp\submit_job.ps1 -Step check_gpu -Dry   # only show the command
#   powershell -ExecutionPolicy Bypass -File .\rcp\submit_job.ps1 -Step check_gpu
#
# Steps and what they reserve (GPU steps land on A100 80GB: the default pool; run_step.sh double-checks).
# Run in this order; steps on the same line can run at the same time (3 GPUs = your guaranteed quota):
#   download_models      0 GPU   CPU only; fetches openai/gpt-oss-120b (~65 GB) if missing
#   check_gptoss         1 GPU: loads gpt-oss-120b and tests the answer parser (run this before the rest)
#   repair_candidates    1 GPU   TranslateGemma: new candidates for the flagged strings   \ together
#   repair_postedit      2 GPU   Qwen3-32B: post-edits of the flagged strings             /
#   repair_collect       0 GPU   merge originals + candidates
#   repair_judge_qwen    2 GPU   judge all candidates (Qwen3-32B)                         \ together
#   repair_judge_gptoss  1 GPU   judge all candidates (gpt-oss-120b)                      /
#   repair_select        0 GPU   pick the best candidate -> work/translations_v2
#   judge_qwen           2 GPU   judge ALL of translations_v2 (Qwen3-32B)                 \ together
#   judge_gptoss         1 GPU   judge ALL of translations_v2 (gpt-oss-120b)              /
#   check_gpu            1 GPU   environment check (already done once)
# Every job ends (and frees its GPUs) as soon as its script finishes. There is no automatic retry (--backoff-limit 0).

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("check_gpu", "download_models", "check_gptoss", "repair_candidates", "repair_postedit", "repair_collect",
                 "repair_judge_qwen", "repair_judge_gptoss", "repair_select", "judge_qwen", "judge_gptoss")]
    [string]$Step,
    [switch]$Dry
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"
Require-Tool runai

$gt = $GptossTp
$plan = @{
    check_gpu           = @{ gpu = 1;   cpu = 4;  mem = "32G" }
    download_models     = @{ gpu = 0;   cpu = 8;  mem = "32G" }
    check_gptoss        = @{ gpu = $gt; cpu = 8;  mem = "128G" }
    repair_candidates   = @{ gpu = 1;   cpu = 8;  mem = "96G" }
    repair_postedit     = @{ gpu = 2;   cpu = 12; mem = "128G" }
    repair_collect      = @{ gpu = 0;   cpu = 2;  mem = "8G" }
    repair_judge_qwen   = @{ gpu = 2;   cpu = 12; mem = "128G" }
    repair_judge_gptoss = @{ gpu = $gt; cpu = 8;  mem = "128G" }
    repair_select       = @{ gpu = 0;   cpu = 2;  mem = "8G" }
    judge_qwen          = @{ gpu = 2;   cpu = 12; mem = "128G" }
    judge_gptoss        = @{ gpu = $gt; cpu = 8;  mem = "128G" }
}
$p = $plan[$Step]
$name = "eedi-" + $Step.Replace("_", "-") + "-" + (Get-Date -Format "MMdd-HHmm")

$cmdArgs = @(
    "submit", "--name", $name, "--project", $Project,
    "-i", $Image, "--image-pull-policy", "IfNotPresent",
    "-g", "$($p.gpu)", "--cpu", "$($p.cpu)", "--memory", $p.mem,
    "--existing-pvc", "claimname=$Pvc,path=$Mount",
    "--run-as-uid", "$RunAsUid", "--run-as-gid", "$RunAsGid", "-e", "EEDI_BASE=$Base", "-e", "EEDI_GPTOSS_TP=$GptossTp",
    "--large-shm", "--backoff-limit", "0",
    "--command", "--", "bash", "$Base/code/rcp/run_step.sh", $Step
)

if ($VllmAttn) { $cmdArgs = $cmdArgs[0..($cmdArgs.Length - 6)] + @("-e", "EEDI_VLLM_ATTN=$VllmAttn") + $cmdArgs[($cmdArgs.Length - 5)..($cmdArgs.Length - 1)] }

Write-Host ""
Write-Host ("Step '{0}': {1} GPU, {2} CPU, {3} RAM" -f $Step, $p.gpu, $p.cpu, $p.mem)
Write-Host ("runai " + ($cmdArgs -join " ")) -ForegroundColor Cyan
if ($Dry) { Write-Host "(dry run: nothing submitted)"; exit 0 }

Run-Native runai $cmdArgs

Write-Host ""
Write-Host "Submitted '$name'. Useful commands:"
Write-Host "  runai list jobs -p $Project                 # status (Pending -> Running -> Succeeded/Failed)"
Write-Host "  runai logs $name -p $Project                # output so far (the same text is saved under $Base/logs/)"
Write-Host "  runai delete job $name -p $Project          # stop it / remove it from the list"
