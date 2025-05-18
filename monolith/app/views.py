from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.urls import reverse
from django.contrib.auth import get_user_model

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
        return render(request, 'home-loggedin.html', {'logout_url': logout_url})
    else:
        login_url = reverse('login')
        return HttpResponse(f"hello stranger <a href='{login_url}'>login</a>")

def logout_view(request):
    logout(request)
    return redirect('home')