from celery import Celery

app = Celery('myproject')

@app.task
def process_audio_frame(audio_frame_id):
    # Logic to process audio frame
    pass