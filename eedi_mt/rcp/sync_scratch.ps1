# Copies files between this PC and the cluster's scratch volume, through a small CPU-ONLY helper job
# (no GPU is used). The helper job is started, used, and deleted again by this script.
#
#   up    : packs the eedi_mt folder (code, rcp\, work\; NOT data\) and unpacks it to /scratch/chafai/eedi_mt/code
#   down  : brings work\ (translations, judgments, pilot results) and the job logs back to this PC
#   -SetToken : (with up) asks for your Hugging Face read token (hidden) and stores it on the volume, readable only by you.
#               The token never appears in a command line or in a job definition.
#
#   powershell -ExecutionPolicy Bypass -File .\rcp\sync_scratch.ps1 -Direction up -SetToken
#   powershell -ExecutionPolicy Bypass -File .\rcp\sync_scratch.ps1 -Direction up
#   powershell -ExecutionPolicy Bypass -File .\rcp\sync_scratch.ps1 -Direction down

param(
    [Parameter(Mandatory = $true)][ValidateSet("up", "down")][string]$Direction,
    [switch]$SetToken,
    [switch]$KeepHelper
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"
Require-Tool runai
Require-Tool kubectl
Require-Tool tar

Write-Host "Starting the CPU-only helper job '$HelperJob' (no GPU) ..."
Run-Native runai @("delete", "job", $HelperJob, "--project", $Project) -AllowFail   # remove a leftover one; an error here is fine
# deleting is asynchronous: wait until the old helper pod is really gone, or the new submit fails with "already exists"
for ($i = 0; $i -lt 24; $i++) {
    $left = & kubectl get pods -n $Namespace -o name 2>$null | Where-Object { $_ -like "pod/$HelperJob-*" }
    if (-not $left) { break }
    Write-Host "  waiting for the previous helper to disappear ..."
    Start-Sleep -Seconds 5
}
Start-Sleep -Seconds 10
Run-Native runai @("submit", "--name", $HelperJob, "--project", $Project, "-i", $HelperImage,
    "--image-pull-policy", "IfNotPresent", "--interactive", "--cpu", "2", "--memory", "4G",
    "--existing-pvc", "claimname=$Pvc,path=$Mount", "--run-as-uid", "$RunAsUid", "--run-as-gid", "$RunAsGid",
    "--command", "--", "sleep", "infinity")

Push-Location $env:TEMP     # kubectl cp gets relative local paths: a Windows path like C:\... confuses its pod:path parser
try {
    # wait until the helper pod is Running
    $pod = $null
    for ($i = 0; $i -lt 60; $i++) {
        $names = & kubectl get pods -n $Namespace -o name 2>$null
        $match = $names | Where-Object { $_ -like "pod/$HelperJob-*" } | Select-Object -First 1
        if ($match) {
            $cand = $match -replace '^pod/', ''
            $phase = & kubectl get pod $cand -n $Namespace -o 'jsonpath={.status.phase}' 2>$null
            if ($phase -eq "Running") { $pod = $cand; break }
        }
        Start-Sleep -Seconds 5
    }
    if (-not $pod) { throw "The helper pod did not start within 5 minutes. Check: runai list jobs -p $Project" }
    Write-Host "Helper pod is running: $pod" -ForegroundColor Green

    # Is the volume mounted and writable for the user the job runs as? Fail here with a clear message if not.
    $check = 'id && mkdir -p ' + $Base + '/code && touch ' + $Base + '/.write_test && rm ' + $Base + '/.write_test && echo WRITE_OK'
    Run-Native kubectl @("exec", $pod, "-n", $Namespace, "--", "sh", "-c", $check)

    if ($Direction -eq "up") {
        $tgz = "eedi_upload.tgz"
        Run-Native tar @("-czf", $tgz, "--exclude=__pycache__", "--exclude=*.tgz", "--exclude=*.zip",
            "--exclude=rcp_probe_*", "--exclude=data", "-C", $ProjectRoot, ".")
        Run-Native kubectl @("cp", $tgz, "$Namespace/${pod}:$Base/upload.tgz")
        # unpack, strip Windows line endings from the .sh files, clean up, list the result
        $remote = 'tar -xzf ' + $Base + '/upload.tgz -C ' + $Base + '/code && ' +
                  'find ' + $Base + '/code -name ''*.sh'' -exec sed -i ''s/\r$//'' {} + && ' +
                  'rm -f ' + $Base + '/upload.tgz && ls -la ' + $Base + '/code && ls ' + $Base + '/code/work'
        Run-Native kubectl @("exec", $pod, "-n", $Namespace, "--", "bash", "-c", $remote)
        Remove-Item $tgz -ErrorAction SilentlyContinue

        if ($SetToken) {
            $sec = Read-Host "Paste your Hugging Face READ token (input is hidden)" -AsSecureString
            $plain = [System.Net.NetworkCredential]::new("", $sec).Password
            $remoteTok = 'umask 077; tr -d ''\r\n'' > ' + $Base + '/.hf_token; ls -l ' + $Base + '/.hf_token'
            $plain | & kubectl exec -i $pod -n $Namespace -- bash -c $remoteTok
            if ($LASTEXITCODE -ne 0) { throw "storing the token failed" }
            $plain = $null
            Write-Host "Token stored (permissions should read -rw-------)." -ForegroundColor Green
        }
    }
    else {
        $remote = 'mkdir -p ' + $Base + '/logs && tar -czf ' + $Base + '/results.tgz -C ' + $Base + '/code work -C ' + $Base + ' logs'
        Run-Native kubectl @("exec", $pod, "-n", $Namespace, "--", "bash", "-c", $remote)
        Run-Native kubectl @("cp", "$Namespace/${pod}:$Base/results.tgz", "eedi_results.tgz")
        Run-Native tar @("-xzf", "eedi_results.tgz", "-C", $ProjectRoot)
        Remove-Item "eedi_results.tgz" -ErrorAction SilentlyContinue
        Run-Native kubectl @("exec", $pod, "-n", $Namespace, "--", "rm", "-f", "$Base/results.tgz")
        Write-Host "Results are now in $ProjectRoot\work and $ProjectRoot\logs" -ForegroundColor Green
    }
}
finally {
    Pop-Location
    if (-not $KeepHelper) {
        Write-Host "Deleting the helper job ..."
        try { & runai delete job $HelperJob --project $Project 2>$null | Out-Null } catch { }
    }
}
