# celeryconfig.py

broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'
timezone = 'UTC'

# Schedule the task to run every 5 minutes
beat_schedule = {
    'log-every-5-minutes': {
        'task': 'tasks.log_message',
        'schedule': 60.0,  # 300 seconds = 5 minutes
    },
}
