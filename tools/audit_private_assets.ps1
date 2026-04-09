param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$audioExtensions = @(".wav", ".mp3", ".flac", ".ogg", ".m4a")
$privateRoots = @(
    "shared/audio/speakers",
    "shared/audio/source_clips",
    "shared/audio/speakers_backups",
    "shared/audio/outputs",
    "shared/audio/temp",
    "shared/audio/temp_conversation_segments",
    "shared/audio/uploads"
) | ForEach-Object { [System.IO.Path]::GetFullPath((Join-Path $Root $_)) }

$excludedRoots = @(
    ".venv",
    ".playwright-cli",
    ".pytest_cache",
    ".vscode",
    "__pycache__"
) | ForEach-Object { [System.IO.Path]::GetFullPath((Join-Path $Root $_)) }

Write-Host "Private asset audit root: $Root"
Write-Host ""
Write-Host "Local-only audio folders:"
foreach ($folder in $privateRoots) {
    if (Test-Path $folder) {
        $count = (Get-ChildItem -Path $folder -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host " - $folder ($count files)"
    } else {
        Write-Host " - $folder (missing)"
    }
}

$allAudio = Get-ChildItem -Path $Root -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $audioExtensions -contains $_.Extension.ToLowerInvariant() }

$unexpected = @()
foreach ($file in $allAudio) {
    $fullPath = [System.IO.Path]::GetFullPath($file.FullName)

    $isExcluded = $false
    foreach ($excludedRoot in $excludedRoots) {
        if ($fullPath.StartsWith($excludedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            $isExcluded = $true
            break
        }
    }
    if ($isExcluded) {
        continue
    }

    $isPrivate = $false
    foreach ($privateRoot in $privateRoots) {
        if ($fullPath.StartsWith($privateRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            $isPrivate = $true
            break
        }
    }

    if (-not $isPrivate) {
        $unexpected += $fullPath
    }
}

Write-Host ""
if ($unexpected.Count -eq 0) {
    Write-Host "No unexpected audio files were found outside the local-only audio folders."
    exit 0
}

Write-Host "Unexpected audio files found outside the local-only folders:" -ForegroundColor Yellow
$unexpected | ForEach-Object { Write-Host " - $_" }
exit 1
