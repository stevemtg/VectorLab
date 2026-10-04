$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe') -or -not (Test-Path -LiteralPath '.runtime\llama\llama.dll') -or -not (Test-Path -LiteralPath 'node_modules\vinext')) {
    throw 'Run .\Setup-VectorLab.ps1 first to install this project''s dependencies.'
}
& node.exe scripts/start-local.mjs
if ($LASTEXITCODE -ne 0) { throw "Vector Lab exited with code $LASTEXITCODE" }
