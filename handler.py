import runpod
import torch
import os
import boto3
import requests
import numpy as np
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
from fish_speech.utils.inference import load_checkpoint, generate_tokens, decode_audio

model_manager = None

def init_model():
    global model_manager
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "checkpoints/s1-mini"
    print(f"Loading Baked Model on {device}...")
    model_manager = load_checkpoint(model_path, device)
    print("✅ Model Ready!")

def upload_to_s3(file_path, file_name):
    s3 = boto3.client('s3', 
        aws_access_key_id=os.getenv('S3_KEY'), 
        aws_secret_access_key=os.getenv('S3_SECRET')
    )
    bucket = os.getenv('S3_BUCKET')
    s3.upload_file(file_path, bucket, file_name)
    return f"https://{bucket}.s3.amazonaws.com/{file_name}"

def handler(job):
    global model_manager
    if model_manager is None: init_model()

    try:
        job_input = job.get("input", {})
        text = job_input.get("text", "").strip()
        ref_audio_url = job_input.get("ref_audio_url", "")
        
        if not text or not ref_audio_url:
            return {"status": "error", "message": "Missing input."}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Reference Audio Download
        ref_data = requests.get(ref_audio_url).content
        with open("ref.wav", "wb") as f: f.write(ref_data)
        
        # 2. 100k Text Chunking
        sentences = sent_tokenize(text)
        audio_segments = []
        
        print(f"Processing {len(sentences)} sentences...")
        for i, s in enumerate(sentences):
            tokens = generate_tokens(model=model_manager.llama, text=s, device=device)
            wav_chunk = decode_audio(model_manager.dac, tokens)
            audio_segments.append(wav_chunk)
            
            # Har 10 sentences baad memory saaf karein
            if i % 10 == 0: torch.cuda.empty_cache()

        # 3. Join & Export
        final_wav = np.concatenate(audio_segments)
        int_audio = (final_wav * 32767).astype(np.int16)
        audio_out = AudioSegment(int_audio.tobytes(), frame_rate=44100, sample_width=2, channels=1)
        
        out_file = f"{job['id']}.mp3"
        audio_out.export(out_file, format="mp3", bitrate="192k")
        
        # 4. S3 Upload
        s3_url = upload_to_s3(out_file, out_file)
        return {"status": "success", "s3_url": s3_url}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})