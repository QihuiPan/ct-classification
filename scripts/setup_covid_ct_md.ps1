[CmdletBinding()]
param(
    [string]$StorageRoot = 'E:\Codex\ct-classification',
    [string]$Python = 'E:\Codex\ct-classification\venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$expectedBytes = 11273767727
$expectedMd5 = '7cd2a4fdc7b1348c093b9a384d4b2240'
$downloadUrl = 'https://ndownloader.figshare.com/files/26069987'
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$datasetDirectory = Join-Path $StorageRoot 'datasets\COVID-CT-MD'
$archive = Join-Path $datasetDirectory 'COVID-CT-MD.zip'
$rawDirectory = Join-Path $datasetDirectory 'raw'
$manifest = Join-Path $datasetDirectory 'manifest.csv'

if (-not $datasetDirectory.StartsWith($StorageRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'The dataset directory escaped the requested storage root.'
}
New-Item -ItemType Directory -Force -Path $datasetDirectory,$rawDirectory | Out-Null

if (-not (Test-Path -LiteralPath $archive) -or (Get-Item -LiteralPath $archive).Length -ne $expectedBytes) {
    & "$env:SystemRoot\System32\curl.exe" -L --fail --retry 8 --retry-delay 5 --continue-at - --output $archive $downloadUrl
    if ($LASTEXITCODE -ne 0) { throw "Dataset download failed with exit code $LASTEXITCODE" }
}

$actualBytes = (Get-Item -LiteralPath $archive).Length
if ($actualBytes -ne $expectedBytes) {
    throw "Archive size mismatch. Expected $expectedBytes bytes, found $actualBytes bytes."
}
$actualMd5 = (Get-FileHash -LiteralPath $archive -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $expectedMd5) {
    throw "Archive MD5 mismatch. Expected $expectedMd5, found $actualMd5."
}

$marker = Join-Path $rawDirectory '.extraction-complete'
if (-not (Test-Path -LiteralPath $marker)) {
    & "$env:SystemRoot\System32\tar.exe" -xf $archive -C $rawDirectory
    if ($LASTEXITCODE -ne 0) { throw "Archive extraction failed with exit code $LASTEXITCODE" }
    New-Item -ItemType File -Force -Path $marker | Out-Null
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Training environment was not found: $Python"
}
& $Python (Join-Path $repositoryRoot 'scripts\prepare_covid_ct_md.py') --dataset-root $rawDirectory --output $manifest
if ($LASTEXITCODE -ne 0) { throw "Manifest preparation failed with exit code $LASTEXITCODE" }

Write-Host "COVID-CT-MD is ready at $datasetDirectory"
