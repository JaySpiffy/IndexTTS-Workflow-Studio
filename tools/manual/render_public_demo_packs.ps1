param(
    [string[]]$PackPaths = @(
        "test_scripts/podcast_roundtable_demo_pack.md",
        "test_scripts/audiobook_night_train_demo_pack.md",
        "test_scripts/game_dialogue_checkpoint_breach_pack.md"
    ),
    [string]$ApiBaseUrl = "http://localhost:8001/api",
    [string]$OutputDir = "docs/assets/social/audio",
    [int]$VersionsPerLine = 1,
    [int]$PollSeconds = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-TrimmedValue {
    param([string]$Value)

    if ($null -eq $Value) {
        return $null
    }

    $trimmed = $Value.Trim()
    if ($trimmed.Length -ge 2) {
        if (($trimmed.StartsWith("'") -and $trimmed.EndsWith("'")) -or ($trimmed.StartsWith('"') -and $trimmed.EndsWith('"'))) {
            return $trimmed.Substring(1, $trimmed.Length - 2)
        }
    }

    return $trimmed
}

function Get-MarkdownSectionCodeFence {
    param(
        [string]$Content,
        [string]$Heading,
        [string]$FenceType
    )

    $pattern = '(?s)##\s+{0}\b.*?```{1}\s*(.*?)\s*```' -f [regex]::Escape($Heading), [regex]::Escape($FenceType)
    $match = [regex]::Match($Content, $pattern)
    if (-not $match.Success) {
        throw "Could not find '$Heading' block in markdown."
    }

    return $match.Groups[1].Value.Trim()
}

function Get-MarkdownHeadingTitle {
    param([string]$Content)

    $match = [regex]::Match($Content, "(?m)^#\s+(.+?)\s*$")
    if ($match.Success) {
        return $match.Groups[1].Value.Trim()
    }

    return "Untitled Pack"
}

function Parse-LinePlans {
    param([string]$PlanText)

    $plans = @{}
    $blockMatches = [regex]::Matches($PlanText, "(?ms)^\s*-\s+id:\s*(L\d+)\s*$.*?(?=^\s*-\s+id:\s*L\d+\s*$|\z)")

    foreach ($blockMatch in $blockMatches) {
        $block = $blockMatch.Value
        $id = Get-TrimmedValue $blockMatch.Groups[1].Value
        if (-not $id) {
            continue
        }

        $indexMatch = [regex]::Match($id, "^L(\d+)$")
        if (-not $indexMatch.Success) {
            continue
        }

        $lineIndex = [int]$indexMatch.Groups[1].Value - 1
        $emotionTextMatch = [regex]::Match($block, "(?m)^\s+emotion_text:\s*(.+?)\s*$")
        $emotionWeightMatch = [regex]::Match($block, "(?m)^\s+emotion_weight:\s*(.+?)\s*$")

        $plans[$lineIndex] = @{
            line_index = $lineIndex
            emotion_text = if ($emotionTextMatch.Success) { Get-TrimmedValue $emotionTextMatch.Groups[1].Value } else { $null }
            emotion_weight = if ($emotionWeightMatch.Success) { [double](Get-TrimmedValue $emotionWeightMatch.Groups[1].Value) } else { $null }
        }
    }

    return $plans
}

function Parse-ScriptPack {
    param([string]$Path)

    $content = Get-Content -Raw -Path $Path
    $title = Get-MarkdownHeadingTitle -Content $content
    $scriptText = Get-MarkdownSectionCodeFence -Content $content -Heading "Pasteable Script" -FenceType "text"
    $planText = Get-MarkdownSectionCodeFence -Content $content -Heading "Emotion And Timing Plan" -FenceType "yaml"
    $linePlans = Parse-LinePlans -PlanText $planText

    $lines = @()
    $scriptLines = $scriptText -split "`r?`n" | Where-Object { $_.Trim() }

    for ($i = 0; $i -lt $scriptLines.Count; $i++) {
        $scriptLine = $scriptLines[$i].Trim()
        $match = [regex]::Match($scriptLine, "^(?<speaker>[^:]+):\s*(?<text>.+)$")
        if (-not $match.Success) {
            throw "Invalid script line format in '$Path': $scriptLine"
        }

        $line = [ordered]@{
            speaker = Get-TrimmedValue $match.Groups["speaker"].Value
            text = Get-TrimmedValue $match.Groups["text"].Value
            line_number = $i
        }

        if ($linePlans.ContainsKey($i)) {
            $plan = $linePlans[$i]
            if ($plan.emotion_text) {
                $line["emotion_control_method"] = "from_text"
                $line["emotion_text"] = $plan.emotion_text
            }
            if ($null -ne $plan.emotion_weight) {
                $line["emotion_weight"] = $plan.emotion_weight
            }
        }

        $lines += [pscustomobject]$line
    }

    return [pscustomobject]@{
        path = $Path
        title = $title
        slug = (($title.ToLowerInvariant() -replace "[^a-z0-9]+", "_").Trim("_"))
        lines = $lines
    }
}

function Get-SpeakerCatalog {
    param([string]$BaseUrl)

    $response = Invoke-RestMethod -Uri "$BaseUrl/speakers" -Method Get
    return @($response.speakers)
}

function Resolve-PublicVoiceMap {
    param([object[]]$Speakers)

    $speakerLookup = @{}
    $availableByName = @{}
    foreach ($speaker in $Speakers) {
        $speakerLookup[$speaker.name.ToLowerInvariant()] = $speaker.filename
        $speakerLookup[$speaker.filename.ToLowerInvariant()] = $speaker.filename
        $availableByName[$speaker.name.ToLowerInvariant()] = $speaker
    }

    $preferredVoiceNames = [ordered]@{
        SpeakerOne = @('Asmongold', 'JoeRogan', 'JohnnyDepp')
        SpeakerTwo = @('ElonMusk', 'JohnnyDepp', 'Asmongold')
        SpeakerThree = @('JeanLucPicard', 'gordeylaforge', 'kajsa')
        SpeakerFour = @('JoeRogan', 'Asmongold', 'ElonMusk')
        SpeakerFive = @('JohnnyDepp', 'JoeRogan', 'Asmongold')
        SpeakerSix = @('Pr.D.Trump', 'ElonMusk', 'JoeRogan')
        SpeakerSeven = @('gordeylaforge', 'JeanLucPicard', 'JohnnyDepp')
        SpeakerEight = @('kajsa', 'JeanLucPicard', 'JoeRogan')
    }

    $fallbackSpeakers = @($Speakers | Sort-Object name)
    $fallbackIndex = 0
    $assignments = @()

    foreach ($alias in $preferredVoiceNames.Keys) {
        $selectedSpeaker = $null

        foreach ($candidateName in $preferredVoiceNames[$alias]) {
            $candidateKey = $candidateName.ToLowerInvariant()
            if ($availableByName.ContainsKey($candidateKey)) {
                $selectedSpeaker = $availableByName[$candidateKey]
                break
            }
        }

        if ($null -eq $selectedSpeaker) {
            if (-not $fallbackSpeakers.Count) {
                throw "No local speakers are available."
            }
            $selectedSpeaker = $fallbackSpeakers[$fallbackIndex % $fallbackSpeakers.Count]
            $fallbackIndex += 1
        }

        $speakerLookup[$alias.ToLowerInvariant()] = $selectedSpeaker.filename
        $speakerLookup[("{0}.wav" -f $alias).ToLowerInvariant()] = $selectedSpeaker.filename

        $assignments += [pscustomobject]@{
            public_label = $alias
            actual_voice = $selectedSpeaker.name
            filename = $selectedSpeaker.filename
        }
    }

    return [pscustomobject]@{
        speaker_lookup = $speakerLookup
        public_voice_assignments = $assignments
    }
}

function Get-PackRenderProfile {
    param([pscustomobject]$Pack)

    if ($Pack.slug -match "audiobook") {
        return @{
            scene_pacing_profile = "balanced"
            scene_gap_ms = 240
        }
    }

    if ($Pack.slug -match "game") {
        return @{
            scene_pacing_profile = "snappy"
            scene_gap_ms = 110
        }
    }

    return @{
        scene_pacing_profile = "balanced"
        scene_gap_ms = 160
    }
}

function Convert-ToConversationPayload {
    param(
        [pscustomobject]$Pack,
        [hashtable]$SpeakerLookup,
        [int]$VersionsPerLineCount
    )

    $renderProfile = Get-PackRenderProfile -Pack $Pack
    $scriptLines = @()

    foreach ($line in $Pack.lines) {
        $speakerKey = $line.speaker.ToLowerInvariant()
        if (-not $SpeakerLookup.ContainsKey($speakerKey)) {
            throw "Speaker '$($line.speaker)' from '$($Pack.path)' was not found in the local voice library."
        }

        $payloadLine = [ordered]@{
            speaker_filename = $SpeakerLookup[$speakerKey]
            text = $line.text
            line_number = $line.line_number
        }

        if ($line.PSObject.Properties.Name -contains "emotion_control_method") {
            $payloadLine["emotion_control_method"] = $line.emotion_control_method
        }
        if ($line.PSObject.Properties.Name -contains "emotion_text") {
            $payloadLine["emotion_text"] = $line.emotion_text
        }
        if ($line.PSObject.Properties.Name -contains "emotion_weight") {
            $payloadLine["emotion_weight"] = $line.emotion_weight
        }

        $scriptLines += $payloadLine
    }

    return @{
        script = @{
            title = $Pack.title
            lines = $scriptLines
        }
        versions_per_line = $VersionsPerLineCount
        similarity_threshold = 0.66
        robotic_threshold = 0.70
        auto_regen_attempts = 0
        seed_strategy = "fixed_base_reused_list"
        fixed_base_seed = 2408
        pacing_preset = "natural"
        scene_pacing_profile = $renderProfile.scene_pacing_profile
        scene_gap_ms = $renderProfile.scene_gap_ms
        respect_punctuation_pauses = $true
        emotion_control_method = "from_speaker"
        emotion_weight = 1.0
        use_random_sampling = $false
        max_text_tokens_per_segment = 100
        do_sample = $false
        top_p = 0.8
        top_k = 30
        temperature = 0.8
        length_penalty = 0.0
        num_beams = 3
        repetition_penalty = 10.0
        max_mel_tokens = 1500
    }
}

function Wait-ConversationCompletion {
    param(
        [string]$BaseUrl,
        [string]$ConversationId,
        [int]$PollDelaySeconds
    )

    while ($true) {
        $status = Invoke-RestMethod -Uri "$BaseUrl/conversation/status/$ConversationId" -Method Get
        $taskStatus = $status.task.status
        $progress = [double]$status.task.progress_percent
        $step = $status.task.current_step
        Write-Host ("[{0}] {1}% - {2}" -f $ConversationId.Substring(0, 8), [math]::Round($progress, 1), $step)

        if ($taskStatus -eq "completed") {
            return $status.task
        }
        if ($taskStatus -in @("failed", "stopped")) {
            throw ("Conversation {0} ended with status '{1}': {2}" -f $ConversationId, $taskStatus, $status.task.error)
        }

        Start-Sleep -Seconds $PollDelaySeconds
    }
}

function Wait-ConcatenationCompletion {
    param(
        [string]$BaseUrl,
        [string]$ConversationId,
        [int]$PollDelaySeconds
    )

    while ($true) {
        $status = Invoke-RestMethod -Uri "$BaseUrl/conversation/status/$ConversationId" -Method Get
        $result = $status.task.result

        if ($null -ne $result -and $result.concatenation_completed) {
            return $status.task
        }
        if ($null -ne $result -and $result.concatenation_error) {
            throw ("Concatenation failed for {0}: {1}" -f $ConversationId, $result.concatenation_error)
        }

        Start-Sleep -Seconds $PollDelaySeconds
    }
}

function Write-Utf8NoBomFile {
    param(
        [string]$Path,
        [string]$Content
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Get-ContainerPathForHostFile {
    param(
        [string]$HostPath,
        [string]$RepoRoot
    )

    $relativePath = [System.IO.Path]::GetRelativePath($RepoRoot, $HostPath)
    return "/app/{0}" -f ($relativePath -replace '\\', '/')
}

function Convert-WavToMp3 {
    param(
        [string]$WavPath,
        [string]$Mp3Path,
        [string]$RepoRoot
    )

    if (Test-Path -LiteralPath $Mp3Path) {
        Remove-Item -LiteralPath $Mp3Path -Force
    }

    $hostFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($hostFfmpeg) {
        & $hostFfmpeg.Source -y -hide_banner -loglevel error -i $WavPath -codec:a libmp3lame -q:a 2 $Mp3Path
        if ($LASTEXITCODE -ne 0) {
            throw "ffmpeg failed while converting '$WavPath' to MP3."
        }
        return
    }

    $containerWavPath = Get-ContainerPathForHostFile -HostPath $WavPath -RepoRoot $RepoRoot
    $containerMp3Path = Get-ContainerPathForHostFile -HostPath $Mp3Path -RepoRoot $RepoRoot

    & docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml exec -T backend `
        ffmpeg -y -hide_banner -loglevel error -i $containerWavPath -codec:a libmp3lame -q:a 2 $containerMp3Path

    if ($LASTEXITCODE -ne 0) {
        throw "Container ffmpeg failed while converting '$WavPath' to MP3."
    }
}

$repoRootPath = (Get-Location).Path
$resolvedOutputDir = Join-Path (Get-Location) $OutputDir
New-Item -Path $resolvedOutputDir -ItemType Directory -Force | Out-Null

$speakers = Get-SpeakerCatalog -BaseUrl $ApiBaseUrl
$voiceResolution = Resolve-PublicVoiceMap -Speakers $speakers
$speakerLookup = $voiceResolution.speaker_lookup
$renderSummary = @()

foreach ($packPath in $PackPaths) {
    Write-Host ""
    Write-Host "=== Rendering $packPath ==="

    $fullPackPath = Join-Path (Get-Location) $packPath
    $pack = Parse-ScriptPack -Path $fullPackPath
    $payload = Convert-ToConversationPayload -Pack $pack -SpeakerLookup $speakerLookup -VersionsPerLineCount $VersionsPerLine
    $jsonBody = $payload | ConvertTo-Json -Depth 8

    $generateResponse = Invoke-RestMethod -Uri "$ApiBaseUrl/conversation/generate" -Method Post -ContentType "application/json" -Body $jsonBody
    $conversationId = $generateResponse.conversation_id
    Write-Host "Started conversation $conversationId"

    $completedStatus = Wait-ConversationCompletion -BaseUrl $ApiBaseUrl -ConversationId $conversationId -PollDelaySeconds $PollSeconds
    $lineCount = @($completedStatus.result.lines).Count

    for ($lineIndex = 0; $lineIndex -lt $lineCount; $lineIndex++) {
        Invoke-RestMethod -Uri "$ApiBaseUrl/conversation/results/$conversationId/line/$lineIndex/select-version?version_index=0" -Method Post | Out-Null
    }

    Invoke-RestMethod -Uri "$ApiBaseUrl/conversation/results/$conversationId/concatenate" -Method Post | Out-Null
    $concatStatus = Wait-ConcatenationCompletion -BaseUrl $ApiBaseUrl -ConversationId $conversationId -PollDelaySeconds 2

    $wavPath = Join-Path $resolvedOutputDir ("{0}.wav" -f $pack.slug)
    $mp3Path = Join-Path $resolvedOutputDir ("{0}.mp3" -f $pack.slug)
    $jsonPath = Join-Path $resolvedOutputDir ("{0}.json" -f $pack.slug)

    Invoke-WebRequest -Uri "$ApiBaseUrl/conversation/results/$conversationId/download" -OutFile $wavPath
    Convert-WavToMp3 -WavPath $wavPath -Mp3Path $mp3Path -RepoRoot $repoRootPath
    Remove-Item -LiteralPath $wavPath -Force

    $metadata = [pscustomobject]@{
        title = $pack.title
        source_pack = $pack.path
        conversation_id = $conversationId
        total_lines = $pack.lines.Count
        audio_path = $mp3Path
        audio_format = "mp3"
        public_voice_assignments = $voiceResolution.public_voice_assignments
        task = $concatStatus
    }
    Write-Utf8NoBomFile -Path $jsonPath -Content ($metadata | ConvertTo-Json -Depth 10)

    $renderSummary += [pscustomobject]@{
        title = $pack.title
        conversation_id = $conversationId
        audio_path = $mp3Path
        mp3_path = $mp3Path
        metadata_path = $jsonPath
    }

    Write-Host "Saved audio to $mp3Path"
}

$summaryPath = Join-Path $resolvedOutputDir "render_summary.json"
Write-Utf8NoBomFile -Path $summaryPath -Content ($renderSummary | ConvertTo-Json -Depth 8)

Write-Host ""
Write-Host "=== Render Summary ==="
$renderSummary | Format-Table -AutoSize
Write-Host "Saved summary to $summaryPath"
