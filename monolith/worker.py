import django
django.setup()
from celery import Celery, shared_task
import numpy as np
import torch
import math
from transformers import pipeline
from app.models import AudioFrame, Transcription
import time

app = Celery('myproject')

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu", torch_dtype=torch.float32)
# bigpipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cpu", torch_dtype=torch.float32)

mu6palette = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
mu6palette_to_int = np.zeros((128,), dtype=np.float32) / 0.0 # all nans, for error detection
for i, c in enumerate(mu6palette):
    mu6palette_to_int[c] = i

def mu6decode(x_mu):
    x = ((x_mu / 63) * 2.0) - 1.0
    x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(63)) - 1.0) / 63
    return x

def mu_string_to_float32(mu_string):
    mu_string = np.array(list(mu_string), dtype=np.int32)
    # now that we have a string of bytes, index into the palette for each byte
    # print(mu_string, mu6palette_to_int)
    mu_bytes = mu6palette_to_int[mu_string]
    mu_floats = mu6decode(mu_bytes)
    # if any mu_floats are NaN, return None
    if np.any(np.isnan(mu_floats)):
        return None
    return mu_floats

@shared_task
def transcribe_audio_frame(audio_frame_id):
    time_start = time.time()
    try:
        audio_frame = AudioFrame.objects.get(id=audio_frame_id)
    except AudioFrame.DoesNotExist:
        return "AudioFrame not found"
    
    # if this isn't the most recent audio frame for this conversation_id of this audio frame, skip it and hope we catch up
    if audio_frame != AudioFrame.objects.filter(
        conversation_id=audio_frame.conversation_id,
    ).order_by('-client_timestamp').first():
        # create a null transcription
        Transcription.objects.create(
            audio_frame=audio_frame,
            fast_transcription=None,
            slow_transcription=None,
        )
        return "Not the most recent audio frame for this conversation_id"


    # get the last sixty audio frames before the current audio_frame's client_timestamp
    audio_frames = AudioFrame.objects.filter(
        user=audio_frame.user,
        client_timestamp__lt=audio_frame.client_timestamp,
    ).order_by('-client_timestamp')[:60]
    if not audio_frames or len(audio_frames) == 0:
        return "No previous audio frames found" 
    # 
    audio_frames = reversed(audio_frames)

    audio_data = [a.audio_data for a in audio_frames]
    audio_data = b''.join(audio_data)
    audio_data = mu_string_to_float32(audio_data)
    if audio_data is None:
        return "Decoding error: NaN values found in audio data"
    time_audio_data_decoded = time.time()

    transcription_result = pipe(audio_data)['text']
    time_audio_data_transcribed = time.time()
    # big_transcription_result = bigpipe(audio_data)
    print("Transcription result:", transcription_result)
    
    # For demo, use the same result for both fast and slow transcriptions.
    Transcription.objects.create(
        audio_frame=audio_frame,
        fast_transcription=str(transcription_result),
        slow_transcription=None, # str(big_transcription_result),
    )
    time_transcript_created = time.time()
    print("Transcription time was", int(1000 * (time_audio_data_transcribed - time_audio_data_decoded)), "ms of", int(1000 * (time_transcript_created - time_start)), "ms total")
    return "Transcription completed"