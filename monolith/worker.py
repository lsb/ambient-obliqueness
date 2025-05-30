import torch
from transformers import pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
fastasr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device, torch_dtype=torch.float32)
slowasr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=device, torch_dtype=torch.float32)

# New: text generation pipeline for summarization
from transformers import pipeline as text_pipeline
summarizer = text_pipeline("text-generation", model="Qwen/Qwen3-0.6B", device=device, torch_dtype=torch.float32)

import django
django.setup()
from celery import Celery, shared_task
import numpy as np
import math
from app.models import Transcription, ConversationAnalysis, AudioFrame
import json
import datetime

import time

app = Celery('myproject', broker='redis://localhost:6379/0')

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
            fast_transcription="",
            slow_transcription="",
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
    print("Transcription result:", transcription_result)
    
    # For demo, use the same result for both fast and slow transcriptions.
    Transcription.objects.create(
        audio_frame=audio_frame,
        fast_transcription=str(transcription_result),
        slow_transcription="",
    )
    time_transcript_created = time.time()
    print("Transcription time was", int(1000 * (time_audio_data_transcribed - time_audio_data_decoded)), "ms of", int(1000 * (time_transcript_created - time_start)), "ms total")
    
    # Only trigger a new summary if no summary exists for audio frames in the last 10 seconds
    time_threshold = audio_frame.client_timestamp - datetime.timedelta(seconds=10)
    recent_summaries = ConversationAnalysis.objects.filter(
        conversation_id=audio_frame.conversation_id,
        analysis_type="summary",
        audio_frame__client_timestamp__gte=time_threshold,
        audio_frame__client_timestamp__lte=audio_frame.client_timestamp,
    )
    if recent_summaries.count() == 0:
        summarize_conversation.delay(str(audio_frame.conversation_id))
    
    return "Transcription completed"


@shared_task
def summarize_conversation(conversation_id):
    transcriptions = Transcription.objects.filter(
        audio_frame__conversation_id=conversation_id,
        fast_transcription__isnull=False
    ).order_by('-audio_frame__client_timestamp')
    # Select one out of every 20 non-null transcriptions (up to 1000)
    selected = reversed(transcriptions[::20][:1000])
    texts = [t.fast_transcription for t in selected]
    if not texts:
        return "No valid transcriptions for summary"
    # create a json blob of the texts, to avoid injection attacks
    text_json = json.dumps(texts, ensure_ascii=False)
    query = "".join([
        "This conversation is currently being transcribed, and I want to summarize it, in JSON format.",
        "The format of this conversation transcript is thirty second snippets that overlap. The audio has been truncated, so the words are more accurate in the middle than at the start and end of each snippet.",
        "Please think about the conversation as a whole, and then summarize. These are the conversation fragments, as a JSON array:",
        "\n","\n",
        json.dumps(texts, ensure_ascii=False),
        "\n","\n",
        "Please only return the JSON object, with a single key 'summary' that contains the summary of the conversation, with no other text or formatting.",
    ])
    summary_result = summarizer(
        [{"role": "user", "content": query}, {"role": "assistant", "content": '{"summary":'}],
        max_new_tokens=40000,
    )[0]['generated_text'][-1]['content']
    print(f"Summary result: {summary_result}")
    latest_audio_frame = transcriptions.first().audio_frame if transcriptions.exists() else None
    ConversationAnalysis.objects.update_or_create(
        conversation_id=conversation_id,
        analysis_type="summary",
        defaults={"audio_frame": latest_audio_frame, "analysis": summary_result}
    )
    return "Summary analysis completed"

