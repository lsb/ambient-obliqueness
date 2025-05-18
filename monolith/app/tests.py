from django.test import TestCase
from django.contrib.auth import get_user_model
from .models import AudioFrame

class UserModelTest(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='testuser',
            password='testpassword'
        )

    def test_user_creation(self):
        self.assertEqual(self.user.username, 'testuser')
        self.assertTrue(self.user.check_password('testpassword'))

class AudioFrameModelTest(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='testuser',
            password='testpassword'
        )
        self.audio_frame = AudioFrame.objects.create(
            user=self.user,
            audio_data=b'some binary data'
        )

    def test_audio_frame_creation(self):
        self.assertEqual(self.audio_frame.user, self.user)
        self.assertEqual(self.audio_frame.audio_data, b'some binary data')