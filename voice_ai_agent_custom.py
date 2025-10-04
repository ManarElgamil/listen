#!/usr/bin/env python3
"""
Voice AI Agent - Speaker Diarization & Interruption Detection (Custom Implementation)
This script analyzes audio files to detect speakers and interruptions using individual models.
"""

import json
import pandas as pd
import os
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import login
from pyannote.audio import Model, Inference
from pyannote.core import Segment
import torch
import soundfile as sf

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
    print("The pyannote models require authentication.")
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

def load_audio_file(file_path):
    """Load audio file using soundfile"""
    try:
        print(f"📁 Loading audio file: {file_path}")
        waveform, sample_rate = sf.read(file_path)
        print(f"✅ Audio loaded: {len(waveform)} samples at {sample_rate}Hz")
        return waveform, sample_rate
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None, None

def detect_speech_segments(inference, audio_file):
    """Detect speech segments using segmentation model"""
    try:
        print("🎯 Detecting speech segments...")
        segmentation = inference(audio_file)
        print(f"✅ Found {len(segmentation)} speech segments")
        return segmentation
    except Exception as e:
        print(f"❌ Error in speech detection: {e}")
        return None

def analyze_speakers_and_interruptions(segmentation):
    """Analyze speakers and detect interruptions"""
    print("🔍 Analyzing speakers and interruptions...")
    
    # Convert segmentation to list of segments
    segments = []
    for item in segmentation:
        # Each item is a tuple: (Segment, array)
        if isinstance(item, tuple) and len(item) >= 1:
            segment_obj = item[0]  # Get the Segment object
            if hasattr(segment_obj, 'start') and hasattr(segment_obj, 'end'):
                segments.append({
                    'start': segment_obj.start,
                    'end': segment_obj.end,
                    'speaker': f"SPEAKER_{len(segments):02d}"  # Simple speaker assignment
                })
    
    # Track speaking times
    speaking_times = defaultdict(float)
    interruptions = []
    
    print(f"📊 Processing {len(segments)} segments...")
    
    for i, segment in enumerate(segments):
        speaker = segment['speaker']
        duration = segment['end'] - segment['start']
        speaking_times[speaker] += duration
        
        # Check for interruptions (overlapping segments)
        if i > 0:
            prev_segment = segments[i - 1]
            
            # Check for overlap
            if segment['start'] < prev_segment['end']:
                overlap_duration = prev_segment['end'] - segment['start']
                interruptions.append({
                    "time": round(segment['start'], 2),
                    "interrupter": speaker,
                    "interrupted": prev_segment['speaker'],
                    "overlap_duration": round(overlap_duration, 2)
                })
    
    return speaking_times, interruptions

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
    
    # Load segmentation model
    try:
        print("Loading segmentation model...")
        model = Model.from_pretrained("pyannote/segmentation-3.0")
        inference = Inference(model)
        print("✅ Segmentation model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading segmentation model: {e}")
        return
    
    # Input audio file
    AUDIO_FILE = "amicorpus/ES2016b/audio/ES2016b.Mix-Lapel.wav"
    
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        return
    
    # Load audio file
    waveform, sample_rate = load_audio_file(AUDIO_FILE)
    if waveform is None:
        return
    
    # Detect speech segments
    segmentation = detect_speech_segments(inference, AUDIO_FILE)
    if segmentation is None:
        return
    
    # Analyze speakers and interruptions
    speaking_times, interruptions = analyze_speakers_and_interruptions(segmentation)
    
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
