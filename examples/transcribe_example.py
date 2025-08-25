#!/usr/bin/env python3
"""
Example script demonstrating how to use the ASR server
"""

import requests
import os

def transcribe_audio_file(audio_file_path, api_key, model_name="nvidia/canary-1b-v2"):
    """
    Transcribe an audio file using the ASR server
    
    Args:
        audio_file_path (str): Path to the audio file to transcribe
        api_key (str): API key for authentication
        model_name (str): Model to use for transcription
    
    Returns:
        dict: Transcription result
    """
    url = "http://localhost:8000/v1/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    with open(audio_file_path, "rb") as f:
        files = {
            "file": (os.path.basename(audio_file_path), f, "audio/wav")
        }
        
        data = {
            "model": model_name,
            "source_lang": "es",
            "target_lang": "es"
        }
        
        response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Transcription failed with status {response.status_code}: {response.text}")

if __name__ == "__main__":
    # Example usage
    API_KEY = os.environ.get("INTERNAL_API_KEY", "your-api-key-here")
    
    # Note: You would need an actual audio file to transcribe
    # audio_file = "path/to/your/audio/file.wav"
    # result = transcribe_audio_file(audio_file, API_KEY)
    # print(f"Transcription: {result['text']}")
    
    print("Example script for ASR transcription")
    print("To use this script, uncomment the code and provide a valid audio file path")