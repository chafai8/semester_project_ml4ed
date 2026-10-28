# READ-ONLY probe of your EPFL RCP (Run:ai) environment - Windows PowerShell, runai CLI only (no kubectl).
# Uses the Run:ai "v1" command syntax, which is what the RCP CLI accepts (runai list project, ...).
# It submits no job, allocates no GPU, and prints no tokens.
# Prerequisites: EPFL network/VPN, `runai` on PATH, and you are logged in (`runai login`).
#
# Run from PowerShell (not cmd):
#   powershell -ExecutionPolicy Bypass -File .\probe_cluster.ps1
#   powershell -ExecutionPolicy Bypass -File .\probe_cluster.ps1 -Project ml4ed-chafai
#
# Output: rcp_probe_<timestamp>.txt (also shown in the terminal). Skim it, then paste it to Claude.
# Every command is independent: if one fails or does not exist in your CLI version, the rest still run.

param([string]$Project = "ml4ed-chafai")

$env:SUPPRESS_DEPRECATION_MESSAGE = "true"
$Out = "rcp_probe_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss")
$Max = 150

function Run([string[]]$CmdArgs) {
    Write-Output ""
    Write-Output ("### runai " + ($CmdArgs -join " "))
    try {
        & runai @CmdArgs 2>&1 | Select-Object -First $Max | ForEach-Object { "$_" }
    } catch {
        Write-Output "(error: $_)"
    }
}

if (-not (Get-Command runai -ErrorAction SilentlyContinue)) {
    Write-Output "runai not found on PATH - see rcp\RCP_CONNECT.md section 1"
    exit 1
}

& {
    Write-Output ("# RCP probe  " + (Get-Date -Format "s") + "  host=" + $env:COMPUTERNAME + "  project=" + $Project)

    Write-Output ""
    Write-Output "## 1. CLI version"
    Run @("version")

    Write-Output ""
    Write-Output "## 2. projects and GPU quota"
    Run @("list", "project")
    Run @("describe", "project", $Project)

    Write-Output ""
    Write-Output "## 3. nodes and GPUs: count, type, how many are in use (may be denied for normal users)"
    Run @("top", "node")
    Run @("list", "nodes")

    Write-Output ""
    Write-Output "## 4. what is already running in my project"
    Run @("list", "jobs", "-p", $Project)

    Write-Output ""
    Write-Output "## 5. which submit options does this CLI have? (GPU flags, storage/PVC flags, image flags)"
    Run @("submit", "--help")

    Write-Output ""
    Write-Output "# done"
} 2>&1 | Tee-Object -FilePath $Out

Write-Output ""
Write-Output "Saved to $Out  (it contains your username/project, no secrets - skim it before sharing)"
