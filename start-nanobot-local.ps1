$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$logs = Join-Path $root "logs"
$nanobot = Join-Path $root ".venv\Scripts\nanobot.exe"
$webui = Join-Path $root "webui"

New-Item -ItemType Directory -Force -Path $logs | Out-Null

# Keep local model/API calls on loopback. Some Windows proxy tools advertise
# themselves through system proxy discovery and can make httpx route 127.0.0.1
# through 127.0.0.1:789x, which breaks local vLLM under load.
$env:NO_PROXY = "127.0.0.1,localhost,::1"
$env:no_proxy = "127.0.0.1,localhost,::1"

$bunCommand = Get-Command bun -ErrorAction SilentlyContinue
if ($bunCommand) {
    $bun = $bunCommand.Source
} else {
    $bun = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter bun.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*Oven-sh.Bun*" } |
        Select-Object -First 1 -ExpandProperty FullName
}

if (-not (Test-Path -LiteralPath $nanobot)) {
    throw "nanobot executable not found: $nanobot"
}
if (-not $bun -or -not (Test-Path -LiteralPath $bun)) {
    throw "bun executable not found. Install Bun first, then rerun this script."
}

Start-Process -FilePath $nanobot `
    -ArgumentList @("gateway", "--verbose") `
    -WorkingDirectory $root `
    -RedirectStandardOutput (Join-Path $logs "nanobot-gateway.out.log") `
    -RedirectStandardError (Join-Path $logs "nanobot-gateway.err.log") `
    -WindowStyle Hidden

Start-Process -FilePath $bun `
    -ArgumentList @(".\node_modules\vite\bin\vite.js", "--host", "127.0.0.1") `
    -WorkingDirectory $webui `
    -RedirectStandardOutput (Join-Path $logs "nanobot-webui.out.log") `
    -RedirectStandardError (Join-Path $logs "nanobot-webui.err.log") `
    -WindowStyle Hidden

Write-Host "nanobot gateway: http://127.0.0.1:18790/health"
Write-Host "WebUI dev:       http://127.0.0.1:5173/"
Write-Host "WebUI built-in:  http://127.0.0.1:8765/"
