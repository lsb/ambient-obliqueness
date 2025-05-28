from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from worker import transcribe_audio_frame  # import the Celery task

import datetime
import uuid

from .models import AudioFrame

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
    if request.user.is_authenticated:
        logout_url = reverse('logout')
        return render(
            request,
            'home-loggedin.html',
            {
                'logout_url': logout_url,
                'conversation_id': str(uuid.uuid4()),
            },
            )
    else:
        login_url = reverse('login')
        return HttpResponse(f"hello stranger <a href='{login_url}'>login</a>")

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

