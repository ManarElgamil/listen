import json
import pandas as pd
from collections import defaultdict
from pyannote.audio import Pipeline

# Load diarization pipeline
print("Loading speaker diarization pipeline...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Input audio file
AUDIO_FILE = "amicorpus/ES2016b/audio/ES2016b.Mix-Lapel.wav"

# Run diarization on the whole file
print("Running speaker diarization...")
diarization = pipeline(AUDIO_FILE)

# Store segments
print("Processing diarization results...")
segments = list(diarization.itertracks(yield_label=True))

# Track speaking times
speaking_times = defaultdict(float)

# Track interruptions
interruptions = []

print(f"Found {len(segments)} speech segments")
print("Analyzing interruptions...")

for i in range(len(segments)):
    turn, _, speaker = segments[i]
    speaking_times[speaker] += (turn.end - turn.start)

    if i > 0:
        prev_turn, _, prev_speaker = segments[i - 1]

        # Check for interruption (overlap + different speaker)
        if (turn.start < prev_turn.end) and (speaker != prev_speaker):
            interruptions.append({
                "time": round(turn.start, 2),
                "interrupter": speaker,
                "interrupted": prev_speaker,
                "overlap_duration": round(prev_turn.end - turn.start, 2)
            })

# Format report
report = {
    "total_speakers": len(speaking_times),
    "speaking_times": {spk: round(t, 2) for spk, t in speaking_times.items()},
    "total_interruptions": len(interruptions),
    "interruptions": interruptions
}

# Save as JSON
with open("meeting_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Save interruptions as CSV
if interruptions:
    df = pd.DataFrame(interruptions)
    df.to_csv("interruptions.csv", index=False)
else:
    # Create empty CSV with headers
    pd.DataFrame(columns=["time", "interrupter", "interrupted", "overlap_duration"]).to_csv("interruptions.csv", index=False)

print("✅ Analysis complete!")
print(f"📊 Found {len(speaking_times)} speakers")
print(f"⏱️  Total interruptions detected: {len(interruptions)}")
print("📁 Results saved to 'meeting_report.json' and 'interruptions.csv'")
