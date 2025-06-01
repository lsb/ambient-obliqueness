import torch
# from gpu_models import Inference
import modal
infer = modal.Cls.from_name("lsb-ambient-research-accelerated-models", "Inference")
slowasr = infer.transcribe.remote
summarizer = infer.summarize.remote




import django
django.setup()
from celery import Celery, shared_task
from app.models import Transcription, ConversationAnalysis, AudioFrame
import json
import datetime
import time

# --- Extracted Constants ---
FAST_TRANSCRIPTION_FRAMES = 58
FAST_TRANSCRIPTION_DURATION_SEC = 29

app = Celery('myproject', broker='redis://localhost:6379/0')

mu6palette = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
mu6palette_to_int = torch.zeros((128,), dtype=torch.float32) / 0.0 # all nans, for error detection
for i, c in enumerate(mu6palette):
    mu6palette_to_int[c] = i

mu6int_to_palette = torch.tensor(list(mu6palette))

def mu6decode(x_mu):
    x = ((x_mu / 63) * 2.0) - 1.0
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(torch.tensor(63))) - 1.0) / 63
    return x

def mu_string_to_float32(mu_string):
    mu_string = torch.tensor(list(mu_string), dtype=torch.int32)
    # now that we have a string of bytes, index into the palette for each byte
    mu_bytes = mu6palette_to_int[mu_string]
    mu_floats = mu6decode(mu_bytes)
    if torch.any(torch.isnan(mu_floats)):
        return None
    return mu_floats

def mu6encode(x):
    return torch.floor((torch.sign(x) * torch.log1p(63 * torch.abs(x)) / torch.log1p(torch.tensor(63)) + 1) / 2 * 63 + 0.5).to(torch.int32)

def mu_float32_to_string(mu_floats):
    mu_floats = mu_floats.to(torch.float32)
    mu_bytes = mu6encode(mu_floats)
    mu_string = bytes(mu6int_to_palette[mu_bytes].tolist())
    return mu_string

@shared_task()
def transcribe_audio_frame(audio_frame_id):

    def garbage_silence_detection(mu6):
        return set(mu6) != {84, 85, 86, 87, 88, 89}
    


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
    # Get the last FAST_TRANSCRIPTION_FRAMES frames (58 frames) in ascending order.
    frames = AudioFrame.objects.filter(
        conversation_id=audio_frame.conversation_id
    ).order_by('-client_timestamp')[:FAST_TRANSCRIPTION_FRAMES]
    frames = list(reversed(frames))
    # Concatenate their audio_data as a string.
    audio_data_str = b''.join([frame.audio_data for frame in frames])

    # Run slowasr() on the raw audio data string.
    transcription_result = slowasr(audio_data_str)['text'] if garbage_silence_detection(audio_data_str) else ""
    # Update the transcription record with the result for both fast and slow transcriptions.
    transcription_record.fast_transcription = transcription_result
    transcription_record.slow_transcription = transcription_result
    transcription_record.save()
    # Trigger the summarization task.
    summarize_conversation.delay(audio_frame.id)
    return f"Transcription completed: {transcription_result}, sound levels {len(set(audio_data_str))}"

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
    # Select one out of every 20 transcriptions (up to 1000), then reverse them so oldest come first.
    selected = list(reversed(transcriptions[::20][:1000]))
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
    )[0]['generated_text'][-1]['content']
    print(f"Summary result: {summary_output}, text json of the conversation is {len(text_json)} characters long, with {len(texts)} snippets.")
    new_record.analysis = summary_output
    new_record.save()
    return "Summary analysis completed"

