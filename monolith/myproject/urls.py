from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('home/', views.home_view, name='home'),
    path('api/upload/', views.upload_audio_frame, name='upload_audio_frame'),
    path('api/analysis/', views.conversation_analysis, name='conversation_analysis'),
    path('static/meter-processor.js', views.meter_processor, name='meter_processor'),
    path('', views.home_view, name='home'),
]