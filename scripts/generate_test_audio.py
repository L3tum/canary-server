#!/usr/bin/env python3
"""
Generate test audio files for benchmarking
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf

def generate_sine_wave(frequency, duration, sample_rate=16000):
    """Generate a sine wave"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t)
    return wave

def generate_speech_like_audio(duration, sample_rate=16000):
    """Generate speech-like audio with varying frequencies"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Combine multiple frequencies to simulate speech
    wave = (np.sin(2 * np.pi * 300 * t) * 0.2 +  # Low frequency
            np.sin(2 * np.pi * 600 * t) * 0.3 +  # Mid frequency
            np.sin(2 * np.pi * 1200 * t) * 0.2 + # High frequency
            np.sin(2 * np.pi * 2000 * t) * 0.1 + # Higher frequency
            np.random.normal(0, 0.1, len(t)))    # Add some noise
    
    # Normalize to prevent clipping
    wave = wave / np.max(np.abs(wave))
    
    return wave

def main():
    parser = argparse.ArgumentParser(description="Generate test audio files")
    parser.add_argument("--output", default="test_audio.wav", help="Output file path")
    parser.add_argument("--duration", type=float, default=5.0, help="Audio duration in seconds")
    parser.add_argument("--type", choices=["sine", "speech"], default="speech", 
                        help="Type of audio to generate")
    parser.add_argument("--frequency", type=float, default=440.0, 
                        help="Frequency for sine wave (Hz)")
    
    args = parser.parse_args()
    
    print(f"Generating {args.type} audio file...")
    print(f"Duration: {args.duration} seconds")
    print(f"Output file: {args.output}")
    
    if args.type == "sine":
        print(f"Frequency: {args.frequency} Hz")
        audio = generate_sine_wave(args.frequency, args.duration)
    else:
        print("Generating speech-like audio")
        audio = generate_speech_like_audio(args.duration)
    
    # Convert to 16-bit integers
    audio = (audio * 32767).astype(np.int16)
    
    # Save to file
    sf.write(args.output, audio, 16000)
    
    # Verify the file
    info = sf.info(args.output)
    print(f"File created successfully!")
    print(f"Sample rate: {info.samplerate} Hz")
    print(f"Duration: {info.duration} seconds")
    print(f"Channels: {info.channels}")

if __name__ == "__main__":
    main()