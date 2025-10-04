# Voice AI Agent - Speaker Diarization & Interruption Detection

A Python application that analyzes audio files to detect speakers and interruptions in meetings using advanced AI models.

## Features

- **Speaker Diarization**: Automatically identifies different speakers in audio
- **Interruption Detection**: Detects when one speaker interrupts another
- **Speaking Time Analysis**: Calculates how much time each speaker talks
- **Detailed Reports**: Generates JSON and CSV reports with analysis results

## Requirements

- Python 3.8+
- Hugging Face account with access to pyannote/speaker-diarization model
- FFmpeg (for audio processing)

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install FFmpeg (if not already installed):
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

## Setup

1. **Get Hugging Face Access**:
   - Go to https://huggingface.co/pyannote/speaker-diarization
   - Click "Agree and access repository"
   - Create a token at https://huggingface.co/settings/tokens

2. **Run the Application**:
   ```bash
   python3 voice_ai_agent.py
   ```
   The script will prompt you for your Hugging Face token if needed.

## Usage

Place your audio file in the `amicorpus/ES2016b/audio/` directory and run:

```bash
python3 voice_ai_agent.py
```

The application will:
1. Load the speaker diarization model
2. Analyze the audio file
3. Detect speakers and interruptions
4. Generate detailed reports

## Output Files

- `meeting_report.json`: Complete analysis results
- `interruptions.csv`: Detailed interruption data with timestamps

## Example Output

```json
{
  "total_speakers": 3,
  "speaking_times": {
    "SPEAKER_00": 45.2,
    "SPEAKER_01": 32.8,
    "SPEAKER_02": 18.5
  },
  "total_interruptions": 5,
  "interruptions": [
    {
      "time": 12.3,
      "interrupter": "SPEAKER_01",
      "interrupted": "SPEAKER_00",
      "overlap_duration": 1.2
    }
  ]
}
```

## Troubleshooting

- **Authentication Error**: Make sure you have access to the pyannote model and a valid token
- **Audio Loading Issues**: Ensure your audio file is in a supported format (WAV, MP3, etc.)
- **FFmpeg Issues**: Verify FFmpeg is properly installed and accessible

## License

This project uses the pyannote.audio library, which has its own licensing terms. Please review the pyannote.audio license before use.