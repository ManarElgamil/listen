#!/usr/bin/env python3
"""
Setup script for Hugging Face token
Run this script to set up your Hugging Face token for the voice AI agent
"""

import os

def setup_pyannote_token():
    print("🔑 PYANNOTE_TOKEN Setup")
    print("=" * 40)
    print()
    print("To use the pyannote/speaker-diarization model, you need a Hugging Face token.")
    print("Follow these steps:")
    print()
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token named 'PYANNOTE_TOKEN' (read access)")
    print("3. Copy the token")
    print("4. Accept the terms for pyannote/speaker-diarization at:")
    print("   https://huggingface.co/pyannote/speaker-diarization")
    print()
    
    token = input("Enter your PYANNOTE_TOKEN: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return
    
    if not token.startswith('hf_'):
        print("⚠️  Warning: Hugging Face tokens usually start with 'hf_'. Are you sure this is correct?")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Setup cancelled.")
            return
    
    # Update .env file
    env_content = f"PYANNOTE_TOKEN={token}\n"
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ PYANNOTE_TOKEN saved to .env file")
        print()
        print("🚀 You can now run: python3 voice_ai_agent.py")
    except Exception as e:
        print(f"❌ Error saving token: {e}")
        print("Please manually create a .env file with:")
        print(f"PYANNOTE_TOKEN={token}")

if __name__ == "__main__":
    setup_pyannote_token()
