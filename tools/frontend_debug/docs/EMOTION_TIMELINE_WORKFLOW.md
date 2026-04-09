# IndexTTS2 Emotion Timeline Workflow

## Overview

The Emotion Timeline Workflow is a new feature that allows users to control emotions on a per-line basis in conversation scripts. This provides granular control over the emotional expression of each line of dialogue, enabling more nuanced and natural-sounding conversations.

## Features

### 1. Script Parsing with Emotion Support
- Parses conversation scripts in the format: `speaker_name: dialogue text`
- Automatically assigns default emotion vectors to each line
- Supports multiple speakers with intelligent speaker matching
- Stores both line information and emotion vectors in a structured format

### 2. Visual Timeline Interface
- Displays each conversation line as a visual block on a timeline
- Shows emotion visualization as colored bars for each line
- Interactive timeline blocks that can be clicked to edit emotions
- Responsive design that works on different screen sizes

### 3. Emotion Control Panel
- Appears when a timeline block is selected
- Provides sliders for adjusting emotion intensities
- Includes preset emotion buttons for quick selection
- Real-time emotion visualization updates
- Support for 8 emotion dimensions: happy, sad, angry, afraid, disgusted, melancholic, surprised, calm

### 4. Emotion Vector Storage
- Each line stores its own 8-dimensional emotion vector
- Vectors are automatically updated when users adjust emotions
- Preserves emotion settings throughout the workflow
- Integrates with existing generation system

## How It Works

### Step 1: Script Input
Users input their conversation script in the standard format:
```
Narrator: The city was quiet...
John: I can't believe it's over.
Sarah: (Surprised) You're here! I thought you were...
```

### Step 2: Automatic Parsing
The system parses the script and:
- Identifies speakers and dialogue text
- Matches speakers to available voice files
- Creates a structured conversation object
- Assigns default emotion vectors (calm: [0, 0, 0, 0, 0, 0, 0, 0.8])

### Step 3: Timeline Generation
For each parsed line, the system:
- Creates a visual timeline block
- Displays speaker name and dialogue text
- Shows emotion visualization as colored bars
- Makes blocks interactive for emotion editing

### Step 4: Emotion Control
When users click on a timeline block:
- Emotion control panel appears
- Current emotion vectors are loaded into sliders
- Users can adjust emotions using sliders or presets
- Changes are applied in real-time to the visualization

### Step 5: Audio Generation
During generation:
- Each line's emotion vector is sent to the backend
- The TTS system uses the emotion vectors to generate appropriate audio
- Multiple versions can be generated per line
- Emotion settings are preserved throughout the process

## Emotion Vector Format

Each emotion vector is an 8-dimensional array:
```
[ happy, sad, angry, afraid, disgusted, melancholic, surprised, calm ]
```

Values range from -1 to 1:
- **0**: No emotion
- **1**: Strong positive emotion
- **-1**: Strong negative emotion

### Preset Emotions

- **Neutral**: [0, 0, 0, 0, 0, 0, 0, 0]
- **Happy**: [0.8, 0, 0, 0, 0, 0, 0.2, 0]
- **Sad**: [0, 0.8, 0, 0, 0, 0.3, 0, 0]
- **Angry**: [0, 0, 0.8, 0, 0.2, 0, 0, 0]
- **Surprised**: [0.2, 0, 0, 0, 0, 0, 0.9, 0]
- **Calm**: [0, 0, 0, 0, 0, 0, 0, 0.8]

## Technical Implementation

### Frontend Components

#### 1. ConversationWorkflow Module (`conversationWorkflow.js`)
- **parseScript()**: Enhanced to support emotion vectors
- **generateTimeline()**: Creates visual timeline from parsed script
- **selectTimelineLine()**: Handles line selection and emotion control display
- **applyEmotionToSelectedLine()**: Updates emotion vectors
- **getEmotionColors()**: Converts vectors to visual colors

#### 2. Core Module (`core.js`)
- **loadSpeakers()**: Loads available speaker voices
- **populateSpeakerSelects()**: Populates speaker selection dropdowns

#### 3. UI Utils Module (`uiUtils.js`)
- **showNotification()**: Displays user notifications

#### 4. CSS Styles (`emotion-timeline.css`)
- Timeline block styling
- Emotion control panel styling
- Responsive design
- Dark mode support

### Data Structures

#### Conversation Script Format
```javascript
let conversationScript = [
    {
        line: 1,
        speaker: "Narrator",
        speaker_filename: "narrator.wav",
        text: "The city was quiet...",
        line_number: 0,
        emo_vector: [0, 0, 0, 0, 0, 0, 0, 0.8]
    },
    {
        line: 2,
        speaker: "John",
        speaker_filename: "John.wav",
        text: "I can't believe it's over.",
        line_number: 1,
        emo_vector: [0, 0, 0.5, 0, 0, 0.5, 0, 0]
    },
    {
        line: 3,
        speaker: "Sarah",
        speaker_filename: "Sarah.wav",
        text: "(Surprised) You're here! I thought you were...",
        line_number: 2,
        emo_vector: [0, 0, 0, 0, 0, 0, 0.9, 0]
    }
];
```

#### Parsed Script Format
```javascript
this.parsedScript = {
    title: "Conversation Title",
    lines: conversationScript
};
```

## Integration with Backend

### API Integration
The emotion vectors are sent to the backend during generation:
```javascript
const generationRequest = {
    script: this.parsedScript,
    emotion_control_method: 'from_vectors',
    // ... other parameters
};
```

### Backend Processing
The backend receives emotion vectors for each line and:
- Uses vectors to guide TTS generation
- Generates multiple versions per line
- Maintains emotion consistency across versions
- Provides quality and similarity scores

## User Interface

### Timeline Blocks
- **Header**: Line number and speaker name
- **Content**: Dialogue text and emotion visualization
- **Interaction**: Click to select and edit emotions
- **Visual Feedback**: Hover effects and selection states

### Emotion Control Panel
- **Sliders**: Individual emotion intensity controls
- **Presets**: Quick emotion selection buttons
- **Actions**: Reset and apply buttons
- **Real-time Updates**: Immediate visual feedback

### Responsive Design
- Adapts to different screen sizes
- Mobile-friendly controls
- Touch-enabled interactions
- Optimized layout for various devices

## Testing

### Test File
A comprehensive test file is provided at `test_emotion_timeline_workflow.html` that demonstrates:
- Script parsing functionality
- Timeline generation
- Emotion control interaction
- Real-time updates

### Test Cases
1. **Basic Script Parsing**: Verify script is parsed correctly
2. **Timeline Generation**: Check timeline blocks are created
3. **Emotion Control**: Test emotion adjustment functionality
4. **Data Persistence**: Verify emotion vectors are maintained
5. **Visual Updates**: Check real-time visualization updates

## Future Enhancements

### Potential Improvements
1. **Audio Preview**: Add audio preview for emotion changes
2. **Emotion Interpolation**: Support for emotion transitions between lines
3. **Batch Operations**: Apply emotions to multiple lines at once
4. **Import/Export**: Save and load emotion configurations
5. **Advanced Presets**: More sophisticated emotion combinations

### Backend Integration
1. **Real-time Generation**: Generate audio preview as emotions are adjusted
2. **Emotion Analysis**: Automatic emotion detection from text
3. **Quality Metrics**: Emotion quality assessment
4. **Voice Adaptation**: Dynamic voice adjustment based on emotions

## Troubleshooting

### Common Issues
1. **Timeline Not Showing**: Check if script was parsed successfully
2. **Emotion Changes Not Applying**: Verify apply button was clicked
3. **Speaker Not Found**: Check speaker file names and matching logic
4. **Visual Updates Not Working**: Check CSS styles are loaded

### Debug Information
The system includes extensive debug logging:
- Script parsing progress
- Speaker matching attempts
- Emotion vector updates
- Timeline generation status

## Conclusion

The Emotion Timeline Workflow provides a powerful and intuitive interface for controlling emotions in conversation scripts. It combines visual feedback, precise control, and seamless integration with the existing IndexTTS2 system to enable users to create more expressive and natural-sounding conversations.