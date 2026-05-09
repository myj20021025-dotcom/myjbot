$ErrorActionPreference = "SilentlyContinue"

$root = $PSScriptRoot
$escapedRoot = [Regex]::Escape($root)
$nanobotExe = Join-Path $root ".venv\Scripts\nanobot.exe"

Get-CimInstance Win32_Process |
    Where-Object {
        ($_.Name -in @("nanobot.exe", "python.exe") -and $_.CommandLine -like "*$nanobotExe*gateway*") -or
        ($_.Name -eq "bun.exe" -and $_.CommandLine -like "*node_modules\vite*") -or
        ($_.CommandLine -match $escapedRoot -and $_.CommandLine -like "*node_modules\vite*")
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force
        Write-Host "Stopped $($_.Name) PID $($_.ProcessId)"
    }
