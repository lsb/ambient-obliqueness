import uuid
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    # Extend the default User model if needed
    pass

class AudioFrame(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    audio_data = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)
    client_timestamp = models.DateTimeField(null=True, blank=True)
    conversation_id = models.UUIDField(default=uuid.uuid4, editable=False, db_index=True)

class Transcription(models.Model):
    audio_frame = models.OneToOneField(AudioFrame, on_delete=models.CASCADE, related_name="transcription")
    fast_transcription = models.TextField(null=True,blank=True)
    slow_transcription = models.TextField(null=True,blank=True)

class ConversationAnalysis(models.Model):
    conversation_id = models.UUIDField(editable=False, db_index=True)
    analysis_type = models.CharField(max_length=100)
    audio_frame = models.ForeignKey(AudioFrame, on_delete=models.CASCADE)
    analysis = models.JSONField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.analysis_type} analysis for {self.conversation_id}"

# when changing this file, rerun making migrations
