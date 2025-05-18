from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    # Extend the default User model if needed
    pass

class AudioFrame(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    audio_data = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)