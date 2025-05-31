import torch
from transformers import pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
slowasr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=device, torch_dtype=torch.float32)
fastasr = slowasr # whisper tiny is not accurate enough
fastasr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device, torch_dtype=torch.float32)

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

# --- Extracted Constants ---
FAST_TRANSCRIPTION_FRAMES = 40
FAST_TRANSCRIPTION_DURATION_SEC = 20
SLOW_TRANSCRIPTION_THRESHOLD_SEC = 30
SUMMARY_THRESHOLD_SEC = 5

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
    mu_bytes = mu6palette_to_int[mu_string]
    mu_floats = mu6decode(mu_bytes)
    if np.any(np.isnan(mu_floats)):
        return None
    return mu_floats

@shared_task()
def transcribe_audio_frame(audio_frame_id):
    time_start = time.time()
    try:
        audio_frame = AudioFrame.objects.get(id=audio_frame_id)
    except AudioFrame.DoesNotExist:
        return "AudioFrame not found"
    
    # Immediately create a transcription record with null transcriptions.
    transcription_record = Transcription.objects.create(
        audio_frame=audio_frame,
        fast_transcription=None,
        slow_transcription=None,
    )
    
    # Skip transcription if this isn't the most recent audio frame.
    if audio_frame != AudioFrame.objects.filter(
        conversation_id=audio_frame.conversation_id,
    ).order_by('-client_timestamp').first():
        return "Not the most recent audio frame for this conversation_id"
    
    # FAST TRANSCRIPTION:
    # Get past FAST_TRANSCRIPTION_FRAMES within the last FAST_TRANSCRIPTION_DURATION_SEC seconds.
    time_lower_bound = audio_frame.client_timestamp - datetime.timedelta(seconds=FAST_TRANSCRIPTION_DURATION_SEC)
    fast_frames = AudioFrame.objects.filter(
        user=audio_frame.user,
        client_timestamp__gte=time_lower_bound,
        client_timestamp__lte=audio_frame.client_timestamp,
    ).order_by('-client_timestamp')[:FAST_TRANSCRIPTION_FRAMES]
    if not fast_frames or len(fast_frames) == 0:
        return "No audio frames for fast transcription"
    fast_frames = reversed(fast_frames)
    fast_audio_data = [a.audio_data for a in fast_frames]
    fast_audio_data = b''.join(fast_audio_data)
    fast_audio_data = mu_string_to_float32(fast_audio_data)
    if fast_audio_data is None:
        return "Decoding error: NaN values found in fast audio data"
    time_fast_data_decoded = time.time()
    
    fast_result = fastasr(fast_audio_data)['text']
    time_fast_data_transcribed = time.time()
    print("Fast transcription result:", fast_result)
    
    # Update the current transcription record with fast transcription.
    transcription_record.fast_transcription = str(fast_result)
    transcription_record.save()
    print("Fast transcription time:", int(1000 * (time_fast_data_transcribed - time_fast_data_decoded)), "ms")
    
    # SLOW TRANSCRIPTION:
    # Check if there is any slow transcription in the past SLOW_TRANSCRIPTION_THRESHOLD_SEC seconds.
    recent_slow = Transcription.objects.filter(
        audio_frame__conversation_id=audio_frame.conversation_id,
        slow_transcription__isnull=False,
        audio_frame__client_timestamp__gte=audio_frame.client_timestamp - datetime.timedelta(seconds=SLOW_TRANSCRIPTION_THRESHOLD_SEC)
    ).exists()
    if not recent_slow:
        # Find the oldest audio frame with a null slow transcription in this conversation.
        null_slow_qs = Transcription.objects.filter(
            audio_frame__conversation_id=audio_frame.conversation_id,
            slow_transcription__isnull=True
        ).order_by('audio_frame__client_timestamp')
        if null_slow_qs.exists():
            oldest_ts = null_slow_qs.first().audio_frame.client_timestamp
            # Gather all transcription records from oldest_ts up to current.
            slow_trans_qs = Transcription.objects.filter(
                audio_frame__conversation_id=audio_frame.conversation_id,
                audio_frame__client_timestamp__gte=oldest_ts,
                audio_frame__client_timestamp__lte=audio_frame.client_timestamp,
            ).order_by('audio_frame__client_timestamp')
            # Mark these records' slow transcriptions as empty string to indicate they're being processed.
            slow_trans_qs.update(slow_transcription="")
            # Concatenate the audio data from these audio frames.
            frames_for_slow = AudioFrame.objects.filter(
                conversation_id=audio_frame.conversation_id,
                client_timestamp__gte=oldest_ts,
                client_timestamp__lte=audio_frame.client_timestamp,
            ).order_by('client_timestamp')
            slow_audio_data = [a.audio_data for a in frames_for_slow]
            slow_audio_data = b''.join(slow_audio_data)
            slow_audio_data = mu_string_to_float32(slow_audio_data)
            if slow_audio_data is not None:
                slow_result = slowasr(slow_audio_data, return_timestamps=True)['text']
                print("Slow transcription result:", slow_result)
                # Update all related transcription records with the slow result for both fast and slow.
                slow_trans_qs.update(slow_transcription=slow_result, fast_transcription=slow_result)
            else:
                print("Decoding error: NaN values in slow audio data")

    # Remove the conversation analysis check here.
    # Instead, call summarize_conversation with the audio_frame id.
    summarize_conversation.delay(audio_frame.id)
    
    time_end = time.time()
    print("Total transcription time:", int(1000 * (time_end - time_start)), "ms")
    return "Transcription completed"

@shared_task()
def summarize_conversation(audio_frame_id):
    print(f"Starting summarization for audio frame {audio_frame_id}")
    # Retrieve the current audio frame and its conversation.
    try:
        audio_frame = AudioFrame.objects.get(id=audio_frame_id)
    except AudioFrame.DoesNotExist:
        return "AudioFrame not found in summarization"
    conversation_id = audio_frame.conversation_id

    # Check if a pending conversation analysis already exists.
    pending_analysis = ConversationAnalysis.objects.filter(
        conversation_id=conversation_id,
        analysis_type="summary",
        analysis__isnull=True
    ).exists()
    if pending_analysis:
        print("Pending summary already exists; skipping summarization")
        return "Pending summary already exists; skipping summarization"

    print(f"Starting summary analysis for conversation {conversation_id} with audio frame {audio_frame_id}")
    # Create a new conversation analysis record with an empty analysis.
    new_record = ConversationAnalysis.objects.create(
        conversation_id=conversation_id,
        analysis_type="summary",
        audio_frame=audio_frame,
        analysis=None  # empty analysis to start
    )

    transcriptions = Transcription.objects.filter(
        audio_frame__conversation_id=conversation_id,
        fast_transcription__isnull=False
    ).order_by('-audio_frame__client_timestamp')
    # Select one out of every 5 transcriptions (up to 1000), then reverse them so oldest come first.
    selected = list(reversed(transcriptions[::5][:1000]))
    texts = [t.slow_transcription or t.fast_transcription for t in selected]
    if not texts:
        new_record.delete()  # cleanup and abort if there are no valid transcriptions.
        return "No valid transcriptions for summary"
    text_json = json.dumps(texts, ensure_ascii=False)
    print(f"The text json of the conversation is {len(text_json)} characters long, with {len(texts)} snippets.")
    query = "\n".join([
        "This conversation is currently being transcribed, and I want to summarize it, in JSON format.",
        "The format of this conversation transcript is thirty second snippets that overlap. The audio has been truncated, so the words are more accurate in the middle than at the start and end of each snippet.",
        "Please think about the conversation as a whole, and then summarize.",
        "Please only return the JSON object, with a single key 'summary' that contains the summary of the conversation, with no other text or formatting.",
        "My next message will be the conversation transcript, which is a JSON array of strings.",
    ])
    # Using the summarizer pipeline.
    summary_output = summarizer(
        [
            {"role": "system", "content": "You are a helpful assistant who summarizes conversations in JSON format."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "Got it!"},
            {"role": "user", "content": text_json},
            {"role": "assistant", "content": '{"summary":'}
        ],
        max_new_tokens=40000,
    )[0]['generated_text'][-1]['content']
    print(f"Summary result: {summary_output}, text json of the conversation is {len(text_json)} characters long, with {len(texts)} snippets.")
    new_record.analysis = summary_output
    new_record.save()
    return "Summary analysis completed"

