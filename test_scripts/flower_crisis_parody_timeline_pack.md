# Flower Crisis Parody Timeline Pack

Reusable comedy/parody scene written with generic placeholder voice labels for public use.

Replace the placeholder labels in this pack, such as `SpeakerOne`, `SpeakerTwo`, and `SpeakerSix`, with the exact voice labels from your own `Available Voices` list before you generate it.

This version has been updated to match the newer v2 scripting guidance:

- shorter lines
- cleaner punctuation
- easier overlap planning
- better rhythm for line-by-line generation

## Voices Used

- `SpeakerSix`
- `SpeakerTwo`
- `SpeakerFour`
- `SpeakerThree`
- `SpeakerEight`
- `SpeakerOne`

## Suggested Settings

- Preset: `Clone Fidelity`
- Random sampling: `Off`
- Auto-detect emotions: `Off` if you want to follow the manual plan below
- Similarity threshold: `0.70`
- Auto-regen attempts: `2`

## Pasteable Script

```text
SpeakerSix: We have a tremendous situation.
SpeakerSix: The flowers in the lobby are demanding voting rights.

SpeakerTwo: That is not a bug.
SpeakerTwo: I connected the tulips to a civic engagement prototype.
SpeakerTwo: They are now technically a startup.

SpeakerFour: Jamie, pull up whether a daffodil can legally become a billionaire.

SpeakerThree: This is the least dignified emergency briefing I have ever attended.

SpeakerEight: I tried to water them.
SpeakerEight: One of them asked for dental coverage.

SpeakerOne: Secret note to the normal people.
SpeakerOne: I am disguised as an emotional support fern.
SpeakerOne: The mission has gone extremely stupid.

SpeakerSix: Nobody told me there would be spy shrubbery.
SpeakerSix: I do not like plants with opinions.

SpeakerFour: Hold on.
SpeakerFour: Are we sure that guy is not just a haunted hedge with Wi-Fi?

SpeakerTwo: We can solve this with an app.
SpeakerTwo: The flowers upload grievances.
SpeakerTwo: The public buys premium petals.
SpeakerTwo: Unrest becomes recurring revenue.

SpeakerThree: No.
SpeakerThree: We will not monetize the salad.

SpeakerEight: For the record, the roses are only angry because Joe called them a government mushroom.

SpeakerFour: That was one time.
SpeakerFour: And they were acting weirdly federal.

SpeakerOne: I can stay calm.
SpeakerOne: I can stay invisible.
SpeakerOne: I am the mulch of justice.
SpeakerOne: I am absolutely not starting to cry in this ficus.

SpeakerSix: He is crying.
SpeakerSix: The bush is crying.
SpeakerSix: This meeting has lost discipline.

SpeakerOne: I am not weak.
SpeakerOne: I just did not expect evil horticulture to feel this personal.

SpeakerThree: Someone give the operative a glass of water.
SpeakerThree: And, if possible, a smaller metaphor.

SpeakerEight: New plan.
SpeakerEight: We apologize to the flowers, stop giving them microphones, and nobody unionizes the begonias.

SpeakerTwo: I still think the begonias could carry a subscription tier.

SpeakerFour: Honestly, I would listen to a begonia podcast.
SpeakerFour: If it had a good guest lineup.

SpeakerSix: This is why the chrysanthemums no longer respect us.
```

## Line IDs

- `L01` SpeakerSix
- `L02` SpeakerSix
- `L03` SpeakerTwo
- `L04` SpeakerTwo
- `L05` SpeakerTwo
- `L06` SpeakerFour
- `L07` SpeakerThree
- `L08` SpeakerEight
- `L09` SpeakerEight
- `L10` SpeakerOne
- `L11` SpeakerOne
- `L12` SpeakerOne
- `L13` SpeakerSix
- `L14` SpeakerSix
- `L15` SpeakerFour
- `L16` SpeakerFour
- `L17` SpeakerTwo
- `L18` SpeakerTwo
- `L19` SpeakerTwo
- `L20` SpeakerTwo
- `L21` SpeakerThree
- `L22` SpeakerThree
- `L23` SpeakerEight
- `L24` SpeakerFour
- `L25` SpeakerFour
- `L26` SpeakerOne
- `L27` SpeakerOne
- `L28` SpeakerOne
- `L29` SpeakerOne
- `L30` SpeakerSix
- `L31` SpeakerSix
- `L32` SpeakerSix
- `L33` SpeakerOne
- `L34` SpeakerOne
- `L35` SpeakerThree
- `L36` SpeakerThree
- `L37` SpeakerEight
- `L38` SpeakerEight
- `L39` SpeakerTwo
- `L40` SpeakerFour
- `L41` SpeakerFour
- `L42` SpeakerSix

## Emotion And Timing Plan

Use this as a companion planning block for manual setup or future automation.

```yaml
scene:
  title: "Flower Crisis Parody"
  overlap_policy: explicit_only
  default_gap_ms: 220
  default_duck_db: -5
  max_auto_overlap_ms: 260

lines:
  - id: L01
    speaker: SpeakerSix
    emotion_text: "boastful, irritated"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 180
    allow_overlap: false

  - id: L02
    speaker: SpeakerSix
    emotion_text: "complaining, theatrical"
    emotion_weight: 1.05
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false
    notes: "Let the opening joke land."

  - id: L03
    speaker: SpeakerTwo
    emotion_text: "dry, smug"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 120
    allow_overlap: false

  - id: L04
    speaker: SpeakerTwo
    emotion_text: "calm, tech-bro confidence"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 120
    allow_overlap: false

  - id: L05
    speaker: SpeakerTwo
    emotion_text: "quietly proud"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L06
    speaker: SpeakerFour
    emotion_text: "excited disbelief"
    emotion_weight: 1.0
    start_mode: overlap_previous
    overlap_prev_ms: 140
    duck_prev_db: -3
    fade_in_ms: 30
    allow_overlap: true

  - id: L07
    speaker: SpeakerThree
    emotion_text: "stern, exhausted dignity"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L08
    speaker: SpeakerEight
    emotion_text: "practical, mildly annoyed"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 100
    allow_overlap: false

  - id: L09
    speaker: SpeakerEight
    emotion_text: "deadpan"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L10
    speaker: SpeakerOne
    emotion_text: "whispering, sneaky"
    emotion_weight: 1.1
    start_mode: after_previous
    gap_after_ms: 100
    allow_overlap: false

  - id: L11
    speaker: SpeakerOne
    emotion_text: "trying to stay hidden"
    emotion_weight: 1.1
    start_mode: after_previous
    gap_after_ms: 100
    allow_overlap: false

  - id: L12
    speaker: SpeakerOne
    emotion_text: "worried, comic despair"
    emotion_weight: 1.15
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L13
    speaker: SpeakerSix
    emotion_text: "offended, suspicious"
    emotion_weight: 1.05
    start_mode: overlap_previous
    overlap_prev_ms: 120
    duck_prev_db: -4
    fade_in_ms: 20
    allow_overlap: true

  - id: L14
    speaker: SpeakerSix
    emotion_text: "loud complaint"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 180
    allow_overlap: false

  - id: L15
    speaker: SpeakerFour
    emotion_text: "curious, amused"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 80
    allow_overlap: false

  - id: L16
    speaker: SpeakerFour
    emotion_text: "genuinely asking"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 240
    allow_overlap: false

  - id: L17
    speaker: SpeakerTwo
    emotion_text: "pitching an app"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L18
    speaker: SpeakerTwo
    emotion_text: "sales-pitch confidence"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L19
    speaker: SpeakerTwo
    emotion_text: "matter-of-fact greed"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L20
    speaker: SpeakerTwo
    emotion_text: "shamelessly upbeat"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L21
    speaker: SpeakerThree
    emotion_text: "hard stop"
    emotion_weight: 1.05
    start_mode: after_previous
    gap_after_ms: 120
    allow_overlap: false

  - id: L22
    speaker: SpeakerThree
    emotion_text: "firm disgust"
    emotion_weight: 1.05
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L23
    speaker: SpeakerEight
    emotion_text: "dry explanation"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 240
    allow_overlap: false

  - id: L24
    speaker: SpeakerFour
    emotion_text: "defensive"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 80
    allow_overlap: false

  - id: L25
    speaker: SpeakerFour
    emotion_text: "still confused"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 240
    allow_overlap: false

  - id: L26
    speaker: SpeakerOne
    emotion_text: "trying to be brave"
    emotion_weight: 1.1
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L27
    speaker: SpeakerOne
    emotion_text: "whispering resolve"
    emotion_weight: 1.1
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L28
    speaker: SpeakerOne
    emotion_text: "overdramatic heroism"
    emotion_weight: 1.1
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L29
    speaker: SpeakerOne
    emotion_text: "voice cracking"
    emotion_weight: 1.15
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L30
    speaker: SpeakerSix
    emotion_text: "mocking"
    emotion_weight: 1.0
    start_mode: overlap_previous
    overlap_prev_ms: 150
    duck_prev_db: -5
    fade_in_ms: 25
    allow_overlap: true

  - id: L31
    speaker: SpeakerSix
    emotion_text: "loud disbelief"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 80
    allow_overlap: false

  - id: L32
    speaker: SpeakerSix
    emotion_text: "grand complaint"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 240
    allow_overlap: false

  - id: L33
    speaker: SpeakerOne
    emotion_text: "hurt but defiant"
    emotion_weight: 1.1
    start_mode: after_previous
    gap_after_ms: 100
    allow_overlap: false

  - id: L34
    speaker: SpeakerOne
    emotion_text: "surprised sincerity"
    emotion_weight: 1.1
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L35
    speaker: SpeakerThree
    emotion_text: "compassionate, controlled"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 120
    allow_overlap: false

  - id: L36
    speaker: SpeakerThree
    emotion_text: "bone-tired dry humor"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L37
    speaker: SpeakerEight
    emotion_text: "problem-solving"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L38
    speaker: SpeakerEight
    emotion_text: "deadpan practical"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 240
    allow_overlap: false

  - id: L39
    speaker: SpeakerTwo
    emotion_text: "quietly shameless"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L40
    speaker: SpeakerFour
    emotion_text: "sincerely interested"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 80
    allow_overlap: false

  - id: L41
    speaker: SpeakerFour
    emotion_text: "thoughtful confusion"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L42
    speaker: SpeakerSix
    emotion_text: "closing complaint"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 350
    allow_overlap: false
    notes: "Final punchline. Leave room after the last word."
```

## Quick Reuse Notes

- Paste the `Pasteable Script` block into the main conversation workflow.
- Keep `Use Random Sampling` off if you want stronger voice fidelity.
- Use the YAML plan when building a timeline or overlap test.
- If Joe or Trump starts bulldozing the mix, lower that track to `60%` to `75%` in the timeline editor and preview again.
