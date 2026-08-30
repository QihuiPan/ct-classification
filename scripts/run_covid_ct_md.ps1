[CmdletBinding()]
param(
    [string]$StorageRoot = 'E:\Codex\ct-classification',
    [switch]$SkipValidation
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $StorageRoot 'venv\Scripts\python.exe'
$config = Join-Path $repositoryRoot 'configs\covid_ct_md.yaml'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Training environment was not found: $python"
}

$cacheDirectories = @{
    TEMP = (Join-Path $StorageRoot 'tmp')
    TMP = (Join-Path $StorageRoot 'tmp')
    PIP_CACHE_DIR = (Join-Path $StorageRoot 'pip-cache')
    MPLCONFIGDIR = (Join-Path $StorageRoot 'mpl-cache')
    TORCH_HOME = (Join-Path $StorageRoot 'torch-cache')
    HF_HOME = (Join-Path $StorageRoot 'hf-cache')
}
foreach ($entry in $cacheDirectories.GetEnumerator()) {
    New-Item -ItemType Directory -Force -Path $entry.Value | Out-Null
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
}
$env:PYTHONPATH = Join-Path $repositoryRoot 'src'

if (-not $SkipValidation) {
    & $python (Join-Path $repositoryRoot 'scripts\validate_data.py') --config $config
    if ($LASTEXITCODE -ne 0) { throw "Data validation failed with exit code $LASTEXITCODE" }
}
& $python (Join-Path $repositoryRoot 'scripts\train.py') --config $config
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
