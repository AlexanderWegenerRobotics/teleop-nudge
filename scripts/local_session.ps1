# One local-sim session: starts simulator, local orchestrator and VR interface, waits for Enter,
# stops everything and collects every episode of the session into the dataset folder.
# Run from the conda env that has teleop-analysis installed:  scripts\local_session.ps1 [-OrchEnv intention] [-Dataset ...] [-WithVideo]

param(
    [string]$Projects     = "C:\Users\ceti\Documents\01_projects",
    [string]$Package      = "C:\Users\ceti\Documents\03_experimental\Windows",
    [string]$Dataset      = "C:\Users\ceti\Documents\05_data\debugging",
    [string]$OrchEnv      = "intention",                    # conda env that runs teleop-orchestrator (needs teleop-intent + teleop-nudge installed)
    [string]$AnalysisEnv  = "",                             # conda env with teleop-analysis; empty = the env this script was started from
    [string]$OrchConfig   = "configs\system_local.yaml",
    [switch]$WithVideo,
    [switch]$Attention,
    [switch]$NoCollect
)

$sim   = Join-Path $Projects "teleop-simulator"
$orch  = Join-Path $Projects "teleop-orchestrator"
$exe   = Join-Path $Package  "teleop_vr_interface.exe"
$ifLog = Join-Path $Package  "teleop_vr_interface\Saved\Logs\TeleOp"
$orchCmd    = "conda activate $OrchEnv && python run.py $OrchConfig"
$collectCmd = if ($AnalysisEnv) { "conda activate $AnalysisEnv && teleop-collect" } else { "teleop-collect" }

foreach ($p in @((Join-Path $sim "launch.bat"), (Join-Path $orch "run.py"), $exe)) {
    if (-not (Test-Path $p)) { Write-Host "[session] missing: $p"; exit 1 }
}

Write-Host "[session] simulator"
$simProc = Start-Process cmd -ArgumentList "/k", "`"$sim\launch.bat`"" -WorkingDirectory $sim -PassThru
Start-Sleep -Seconds 4

Write-Host "[session] local orchestrator ($OrchConfig)"
$orchProc = Start-Process cmd -ArgumentList "/k", $orchCmd -WorkingDirectory $orch -PassThru
Start-Sleep -Seconds 3

Write-Host "[session] interface"
$ifProc = Start-Process $exe -WorkingDirectory $Package -PassThru

Write-Host ""
Write-Host "[session] running. Press Enter here to stop all three and collect."
[void](Read-Host)

foreach ($p in @($ifProc, $orchProc, $simProc)) {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
}
Get-Process avatar, avatar_pipeline, teleop_vr_interface* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

if ($NoCollect) { exit 0 }

$orchLogs = Join-Path $orch "logs"
$session = Get-ChildItem $orchLogs -Directory | Where-Object { $_.Name -match '^\d{5,}$' } |
           Sort-Object Name -Descending | Select-Object -First 1
if (-not $session) { Write-Host "[session] no session folder in $orchLogs (avatar episode events not received?)"; exit 1 }
$episodes = Get-ChildItem $session.FullName -Directory | Where-Object { $_.Name -match '^\d{3}$' } | ForEach-Object { [int]$_.Name }
if (-not $episodes) { Write-Host "[session] session $($session.Name) has no episodes"; exit 1 }

$args = @($session.Name) + $episodes + @("--out", $Dataset, "--avatar", (Join-Path $sim "logs"), "--local", $orchLogs, "--interface", $ifLog)
if ($WithVideo) { $args += "--with-video" }
if ($Attention) { $args += "--attention" }
Write-Host "[session] collecting session $($session.Name) episodes $($episodes -join ' ') -> $Dataset"
cmd /c "$collectCmd $($args -join ' ')"
