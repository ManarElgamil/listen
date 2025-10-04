#!/usr/bin/env python3
"""
Voice AI Agent - Speaker Diarization and Interruption Detection
This script analyzes audio files to detect speakers and interruptions.
"""

import json
import pandas as pd
import os
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import login
from pyannote.audio import Pipeline

# Load environment variables
load_dotenv()

def get_pyannote_token():
    """Get PYANNOTE_TOKEN from environment or prompt user"""
    # First try to get from environment variable
    token = os.getenv('PYANNOTE_TOKEN')
    
    if token:
        print("✅ Found PYANNOTE_TOKEN in environment variables")
        return token
    
    # If not found, prompt user
    print("🔑 PYANNOTE_TOKEN Required")
    print("=" * 50)
    print()
    print("The pyannote/speaker-diarization model requires authentication.")
    print("Please follow these steps:")
    print()
    print("1. Go to: https://huggingface.co/pyannote/speaker-diarization")
    print("2. Click 'Agree and access repository'")
    print("3. Go to: https://huggingface.co/pyannote/segmentation")
    print("4. Click 'Agree and access repository'")
    print("5. Go to: https://huggingface.co/settings/tokens")
    print("6. Create a new token named 'PYANNOTE_TOKEN' (read access)")
    print("7. Copy the token")
    print()
    
    token = input("Enter your PYANNOTE_TOKEN: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return None
    
    return token

def main():
    print("🎤 Voice AI Agent - Speaker Diarization & Interruption Detection")
    print("=" * 70)
    print()
    
    # Get PYANNOTE_TOKEN and login
    token = get_pyannote_token()
    if not token:
        return
    
    # Login to Hugging Face Hub
    try:
        print("🔐 Logging in to Hugging Face Hub...")
        login(token=token)
        print("✅ Login successful!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    # Load pipeline with explicit model configuration
    try:
        print("Loading speaker diarization pipeline...")
        # Use the correct segmentation model version
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            segmentation="pyannote/segmentation-3.0"
        )
        print("✅ Pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        print("\nTrying alternative approach...")
        try:
            # Try with explicit token parameter
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                segmentation="pyannote/segmentation-3.0",
                token=token
            )
            print("✅ Pipeline loaded successfully with token!")
        except Exception as e2:
            print(f"❌ Still failing: {e2}")
            print("\nThis might be due to:")
            print("- Invalid or expired token")
            print("- No access to pyannote/speaker-diarization model")
            print("- Network connectivity issues")
            print("- Model version compatibility issues")
            return
    
    # Input audio file
    AUDIO_FILE = "amicorpus/ES2016b/audio/ES2016b.Mix-Lapel.wav"
    
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        return
    
    print(f"📁 Analyzing audio file: {AUDIO_FILE}")
    
    # Run diarization on the whole file
    print("🎯 Running speaker diarization...")
    try:
        diarization = pipeline(AUDIO_FILE)
    except Exception as e:
        print(f"❌ Error during diarization: {e}")
        print("\nThis might be due to:")
        print("- Audio format not supported")
        print("- File corruption")
        print("- Insufficient permissions")
        return
    
    # Store segments
    print("📊 Processing diarization results...")
    segments = list(diarization.itertracks(yield_label=True))
    
    # Track speaking times
    speaking_times = defaultdict(float)
    
    # Track interruptions
    interruptions = []
    
    print(f"Found {len(segments)} speech segments")
    print("🔍 Analyzing interruptions...")
    
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
    
    print()
    print("✅ Analysis complete!")
    print(f"👥 Found {len(speaking_times)} speakers")
    print(f"⏱️  Total interruptions detected: {len(interruptions)}")
    print("📁 Results saved to 'meeting_report.json' and 'interruptions.csv'")
    
    # Display summary
    print()
    print("📋 Summary:")
    for speaker, time in speaking_times.items():
        print(f"   {speaker}: {time:.2f} seconds")
    
    if interruptions:
        print()
        print("🚨 Interruptions detected:")
        for i, interruption in enumerate(interruptions[:5], 1):  # Show first 5
            print(f"   {i}. At {interruption['time']}s: {interruption['interrupter']} interrupted {interruption['interrupted']}")
        if len(interruptions) > 5:
            print(f"   ... and {len(interruptions) - 5} more (see interruptions.csv)")

if __name__ == "__main__":
    main()
