from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from worker import transcribe_audio_frame, summarize_conversation  # import the Celery task

import datetime
import uuid
import json

from .models import AudioFrame, Transcription, ConversationAnalysis

User = get_user_model()

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
    return render(request, 'login.html')

def signup_view(request):
    total_users = User.objects.count()
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username and password:
            try:
                user = User.objects.create_user(username=username, password=password)
                login(request, user)
                return redirect('home')
            except Exception as e:
                return HttpResponse("Signup failed: " + str(e), status=400)
    return render(request, 'signup.html', {'total_users': total_users})

def home_view(request):
    if not request.user.is_authenticated:
        login_url = reverse('login')
        return HttpResponse(f"hello stranger <a href='{login_url}'>login</a>")
    
    conv_id = request.GET.get('conversation_id')
    if not conv_id:
        # No query parameter: create a new conversation UUID and redirect.
        new_conv = str(uuid.uuid4())
        return redirect(reverse('home') + f"?conversation_id={new_conv}")
    
    try:
        conv_uuid = uuid.UUID(conv_id)
    except ValueError:
        # Invalid UUID: generate a new one and redirect.
        new_conv = str(uuid.uuid4())
        return redirect(f"?conversation_id={new_conv}")
    
    # Check if any AudioFrame for this conversation belongs to a different user.
    conflict = AudioFrame.objects.filter(conversation_id=conv_uuid).exclude(user=request.user).exists()
    
    if conflict:
        # If unauthorized, render the unauthorized page.
        logout_url = reverse('logout')
        return HttpResponse(f"this isn't your conversation. <a href='{logout_url}'>log out</a>", context)
    
    context = {
        'logout_url': reverse('logout'),
        'conversation_id': conv_id,
    }
    return render(request, 'home-loggedin.html', context)

def logout_view(request):
    logout(request)
    return redirect('home')

@login_required
def upload_audio_frame(request):
    if request.method != 'POST':
        return HttpResponseBadRequest("Only POST method is allowed.")

    # Expecting audio file in files and client timestamp in POST data
    audio_data = request.POST.get('audio_data').encode('utf-8')
    client_ts_str = request.POST.get('client_timestamp')
    conv_id_str = request.POST.get('conversation_id')

    if not audio_data or not client_ts_str:
        return HttpResponseBadRequest("Missing audio_data or client_timestamp.")

    try:
        client_ts = datetime.datetime.fromisoformat(client_ts_str)
    except ValueError:
        return HttpResponseBadRequest("Invalid client_timestamp format. Use ISO format.")

    try:
        conversation_id = uuid.UUID(conv_id_str)
    except ValueError:
        return HttpResponseBadRequest("Invalid conversation_id format.")

    new_audio_frame = AudioFrame.objects.create(
        user=request.user,
        audio_data=audio_data,
        client_timestamp=client_ts,
        conversation_id=conversation_id,
    )
    
    # Enqueue asynchronous job to transcribe this audio frame
    transcribe_audio_frame.delay(new_audio_frame.id)
    
    return JsonResponse({'status': 'success'})


def meter_processor(request):
    return render(request, 'meter-processor.js', content_type='application/javascript')

@login_required
def conversation_analysis(request):
    conversation_id_str = request.GET.get('conversation_id')
    if not conversation_id_str:
        return HttpResponseBadRequest("Missing conversation_id.")
    try:
        conversation_id = uuid.UUID(conversation_id_str)
    except ValueError:
        return HttpResponseBadRequest("Invalid conversation_id format.")

    transcriptions = Transcription.objects.filter(
        audio_frame__conversation_id=conversation_id
    ).order_by('-audio_frame__client_timestamp')[:60]

    results = []
    for t in transcriptions:
        results.append({
            'audio_frame_id': str(t.audio_frame_id),
            'fast_transcription': t.fast_transcription,
            'slow_transcription': t.slow_transcription,
            'client_timestamp': t.audio_frame.client_timestamp.isoformat(),
        })
    # Include the latest summary analysis in the response, for the latest summary that is not null
    summary_analysis = ConversationAnalysis.objects.filter(
        conversation_id=conversation_id,
        analysis_type="summary",
        analysis__isnull=False
    ).order_by('-id').first()
    analysis = summary_analysis.analysis if summary_analysis else None
    if analysis is None:
        analysis = json.dumps({"summary": "No summary available."})

    print("Analysis:", analysis)
    summary = json.loads(analysis).get("summary") if summary_analysis else ""
    return JsonResponse({'transcriptions': results, 'summary': summary})

