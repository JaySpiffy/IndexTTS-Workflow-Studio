param(
    [Parameter(Mandatory = $true)]
    [string]$BackupFolder
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$speakersDir = Join-Path $repoRoot "shared\audio\speakers"
$backupsRoot = Join-Path $repoRoot "shared\audio\speakers_backups"
$resolvedBackup = Resolve-Path -LiteralPath $BackupFolder -ErrorAction Stop

if (-not (Test-Path -LiteralPath $speakersDir -PathType Container)) {
    throw "Speaker directory not found: $speakersDir"
}

$backupPath = $resolvedBackup.Path
if (-not ($backupPath -like "$backupsRoot*")) {
    throw "Backup path must be inside shared\\audio\\speakers_backups"
}

$wavFiles = Get-ChildItem -LiteralPath $backupPath -Filter *.wav -File
if (-not $wavFiles) {
    throw "No .wav files found in backup folder: $backupPath"
}

foreach ($file in $wavFiles) {
    Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $speakersDir $file.Name) -Force
}

Write-Host "Restored $($wavFiles.Count) speaker files from $backupPath to $speakersDir"
