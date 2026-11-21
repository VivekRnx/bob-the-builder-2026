# Full pre-submit check on Windows. Usage: powershell -File scripts\check.ps1 [-Docker]
param([switch]$Docker)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Step($name, [scriptblock]$cmd) { Write-Host "== $name"; & $cmd; if ($LASTEXITCODE -ne 0) { throw "$name failed" } }
Step "scenario tests"      { python tools/test_brain.py }
Step "latency bench"       { python tools/bench.py }
Step "conformance (local)" { python mechfight-kit/check/check.py -- python -u bot.py }
if ($Docker) {
  Step "docker build" { docker build --platform linux/amd64 -t my-bot . }
  $arena = (Resolve-Path "mechfight-kit/arena").Path
  Step "conformance (match sandbox)" {
    python mechfight-kit/check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 `
      --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "${arena}:/arena:ro" my-bot
  }
}
