# Charter docs/charter-stare-input-representation-v1.md §7-a -- sequential driver for the judged run.
#
# Runs every (seed, arm) pair one after another with the runner's --resume, so after a machine
# reset the SAME command continues where the chain stopped: finished runs (summary.json with
# epochs_done >= Epochs) are skipped, a partial run resumes from its last checkpoint, and a stale
# LONG_RUN.lock whose pid is dead is stolen. A live lock aborts the driver (exit 3).
#
# Each run is a child started with Start-Process (no PowerShell stderr wrapping) that the driver
# waits on while sampling the GPU; its stdout/stderr go to timestamped files so a relaunch never
# truncates an earlier log.
#
#   # start detached through WMI (returns at once; survives the calling app; progress in <OutRoot>\driver.log):
#   powershell -File .harness/scripts/run_stare_judged.ps1 -Detach
#   # after a reset: the same command again.
#
# GPU guard (2026-10-22, after a Windows Update reboot silently dropped the user's GPU caps and the
# machine hard-reset mid-run): before each run the driver waits until nvidia-smi reports
# power.limit <= -MaxPowerW; during the run it samples the SM clock every 15 s and, if it exceeds
# -MaxClockMHz on two consecutive samples (clock lock `-lgc` not applied), it stops the run (the
# runner checkpoints every -Chunk epochs) and exits with code 5. The driver cannot apply the caps
# itself: they need an elevated shell (`nvidia-smi -pl 160`, `nvidia-smi -lgc 300,1400`).
#
# -Module picks the runner (charter-hahe-same-harness-v1 §3): the default StarE bridge, or
#   -Module src.ontology.hahe_bridge.runner -Arms hahe -Epochs 300 -OutRoot outputs\hahe_same_harness_v1\judged
param(
  [string]$Seeds = '1,2,3,4,5',
  [string]$Arms = 'hyper,reified',
  [int]$Epochs = 401,
  [int]$EvalEvery = 10,
  [int]$Chunk = 5,
  [double]$StepSleepMs = 150,
  [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
  [string]$OutRoot = 'outputs\stare_input_representation_v1\judged',
  [double]$MaxPowerW = 160,
  [int]$MaxClockMHz = 1450,
  [switch]$SkipGpuGuard,
  [switch]$Detach,
  [string]$Module = 'src.ontology.stare_bridge.runner'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$py = Join-Path $repo '.venv\Scripts\python.exe'
$root = Join-Path $repo $OutRoot
New-Item -ItemType Directory -Force -Path $root | Out-Null
$driverLog = Join-Path $root 'driver.log'

if ($Detach) {
  $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath,
    '-Seeds', $Seeds, '-Arms', $Arms, '-Epochs', "$Epochs", '-EvalEvery', "$EvalEvery",
    '-Chunk', "$Chunk", '-StepSleepMs', "$StepSleepMs", '-Device', $Device, '-OutRoot', $OutRoot,
    '-MaxPowerW', "$MaxPowerW", '-MaxClockMHz', "$MaxClockMHz", '-Module', $Module)
  if ($SkipGpuGuard) { $argList += '-SkipGpuGuard' }
  # Created through WMI, not Start-Process, so the driver is not a child of the calling app: on
  # 2026-10-28 08:41 a Claude desktop auto-update restart took a Start-Process driver and its run down.
  $cmd = 'powershell ' + (($argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ')
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $repo }
  if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed with $($r.ReturnValue)" }
  Set-Content -Path (Join-Path $root 'driver_pid.txt') -Value $r.ProcessId
  "driver pid=$($r.ProcessId) log=$driverLog"
  return
}

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$lock = Join-Path $repo '.harness\tmp\LONG_RUN.lock'

function Log([string]$msg) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
  Add-Content -Path $driverLog -Value $line
  Write-Output $line
}

function GpuState {
  $line = (& nvidia-smi --query-gpu=power.limit,clocks.sm --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  if (-not $line) { return $null }
  $parts = $line -split ','
  return [pscustomobject]@{ PowerLimitW = [double]$parts[0].Trim(); SmMHz = [int]$parts[1].Trim() }
}

$telemetry = Join-Path $root 'telemetry.csv'
function Telemetry([int]$launcherId) {
  # One fsynced row per minute while a run is live, so a slowdown or a reset can be read back
  # afterwards (2026-10-30: the machine slowed for a while and then hard-reset with nothing logged).
  $g = (& nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  $os = Get-CimInstance Win32_OperatingSystem
  $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
  $runner = Get-CimInstance Win32_Process -Filter "ParentProcessId=$launcherId and Name='python.exe'" | Select-Object -First 1  # venv launcher -> python (not its conhost)
  if (-not $runner) { $runner = Get-CimInstance Win32_Process -Filter "ProcessId=$launcherId" }
  $wsMB = if ($runner) { [int]($runner.WorkingSetSize / 1MB) } else { '' }
  $tcp = @(Get-NetTCPConnection -ErrorAction SilentlyContinue).Count
  $row = '{0},{1},{2},{3},{4},{5},{6}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), ($g -replace ' ', ''), $cpu,
    [int]($os.FreePhysicalMemory / 1024), [int]($os.FreeVirtualMemory / 1024), $wsMB, $tcp
  $header = if (Test-Path $telemetry) { '' } else { "time,gpu_power_w,sm_mhz,gpu_temp_c,gpu_util_pct,gpu_mem_mb,cpu_load_pct,free_ram_mb,free_virtual_mb,runner_ws_mb,tcp_connections`r`n" }
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($header + $row + "`r`n")
  $fs = [System.IO.File]::Open($telemetry, 'Append', 'Write', 'Read')
  try { $fs.Write($bytes, 0, $bytes.Length); $fs.Flush($true) } finally { $fs.Dispose() }
}

function WaitForPowerCap {
  if ($SkipGpuGuard -or $Device -ne 'cuda') { return }
  $announced = $false
  while ($true) {
    $g = GpuState
    if ($g -and $g.PowerLimitW -le $MaxPowerW) { if ($announced) { Log "GPU power limit now $($g.PowerLimitW) W; continuing" }; return }
    if (-not $announced) { Log "waiting: GPU power limit $($g.PowerLimitW) W > $MaxPowerW W. Apply in an elevated shell: nvidia-smi -pl 160 ; nvidia-smi -lgc 300,1400"; $announced = $true }
    Start-Sleep -Seconds 60
  }
}

Log "driver start module=$Module seeds=$Seeds arms=$Arms epochs=$Epochs eval_every=$EvalEvery chunk=$Chunk step_sleep_ms=$StepSleepMs device=$Device max_power_w=$MaxPowerW max_clock_mhz=$MaxClockMHz guard=$(-not $SkipGpuGuard)"
foreach ($seed in ($Seeds -split ',')) {
  foreach ($arm in ($Arms -split ',')) {
    $seed = $seed.Trim(); $arm = $arm.Trim()
    $dir = Join-Path $root (Join-Path $arm ("seed_{0}" -f $seed))
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $summary = Join-Path $dir 'summary.json'
    if (Test-Path $summary) {
      $s = Get-Content $summary -Raw | ConvertFrom-Json
      if ([int]$s.epochs_done -ge $Epochs) { Log "skip $arm seed $seed (epochs_done=$($s.epochs_done))"; continue }
    }
    $args = @('-u', '-m', $Module, '--arm', $arm, '--seed', "$seed",
      '--epochs', "$Epochs", '--eval-every', "$EvalEvery", '--chunk', "$Chunk", '--device', $Device,
      '--out', $dir, '--use-test', '--resume')
    if ($StepSleepMs -gt 0) { $args += @('--step-sleep-ms', "$StepSleepMs") }
    if (Test-Path $lock) {
      $info = Get-Content $lock -Raw | ConvertFrom-Json
      $alive = $null -ne (Get-Process -Id ([int]$info.pid) -ErrorAction SilentlyContinue)
      if ($alive) { Log "abort: LONG_RUN.lock held by live pid $($info.pid)"; exit 3 }
      Log "stale LONG_RUN.lock from pid $($info.pid); passing --steal-lock"
      $args += '--steal-lock'
    }
    WaitForPowerCap
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    Log "start $arm seed $seed -> $dir"
    $p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput (Join-Path $dir "stdout_$stamp.log") -RedirectStandardError (Join-Path $dir "stderr_$stamp.log")
    $null = $p.Handle  # cache the handle so ExitCode is available after exit
    $over = 0
    $tick = 0
    while (-not $p.WaitForExit(15000)) {
      if ($tick++ % 4 -eq 0) { try { Telemetry $p.Id } catch { } }
      if ($SkipGpuGuard -or $Device -ne 'cuda') { continue }
      $g = GpuState
      if ($g -and $g.SmMHz -gt $MaxClockMHz) { $over += 1 } else { $over = 0 }
      if ($over -ge 2 -or ($g -and $g.PowerLimitW -gt $MaxPowerW)) {
        Log "GPU guard: SM clock $($g.SmMHz) MHz / power limit $($g.PowerLimitW) W exceeds the cap ($MaxClockMHz MHz, $MaxPowerW W); stopping $arm seed $seed (resumes from its last checkpoint)"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        $null = $p.WaitForExit(30000)
        Log "driver stopped by the GPU guard. Apply the caps in an elevated shell, then rerun the same command."
        exit 5
      }
    }
    $code = $p.ExitCode
    Log "end $arm seed $seed exit=$code"
    if ($code -ne 0) { Log "stopping the chain on a non-zero exit"; exit $code }
  }
}
Log "driver done"
