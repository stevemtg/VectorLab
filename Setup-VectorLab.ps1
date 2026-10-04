$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

$nodePath = (Get-Command node.exe -ErrorAction Stop).Source
$pythonPath = (Get-Command python.exe -ErrorAction Stop).Source
$npmCli = Join-Path (Split-Path -Parent $nodePath) 'node_modules\npm\bin\npm-cli.js'
Invoke-Checked $nodePath @($npmCli, 'install', '--no-audit', '--no-fund')
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    Invoke-Checked $pythonPath @('-m', 'venv', '.venv')
}
Invoke-Checked (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') @('-m', 'pip', 'install', '-r', 'backend\requirements.txt')

$runtimeDirectory = Join-Path $PSScriptRoot '.runtime\llama'
$archiveFile = Join-Path $PSScriptRoot '.runtime\llama-cpu.zip'
$expectedHash = '14CF1303CA9AC3ABD94816850532F9F9A69AC66FBACA3776FC6F9061C2FAC1D1'
if (-not (Test-Path -LiteralPath $runtimeDirectory)) { New-Item -ItemType Directory -Path $runtimeDirectory | Out-Null }
if (-not (Test-Path -LiteralPath $archiveFile)) {
    Invoke-WebRequest -Uri 'https://github.com/ggml-org/llama.cpp/releases/download/b11146/llama-b11146-bin-win-cpu-x64.zip' -OutFile $archiveFile -UseBasicParsing
}
if ((Get-FileHash -LiteralPath $archiveFile -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'The llama.cpp archive checksum does not match the pinned b11146 build. Installation stopped.'
}
if (-not (Test-Path -LiteralPath (Join-Path $runtimeDirectory 'llama.dll'))) {
    Expand-Archive -LiteralPath $archiveFile -DestinationPath $runtimeDirectory
}
Write-Output 'Setup complete. Run .\Start-VectorLab.ps1 to open the local workspace.'
