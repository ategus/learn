# tasks.py

from celery import Celery
import logging
import datetime

# Create Celery instance
app = Celery('tasks')
app.config_from_object('celeryconfig')

# Set up logging
logging.basicConfig(filename='logs/task.log', level=logging.INFO, format='%(asctime)s - %(message)s')

@app.task
def log_message():
    logging.info("Celery task executed at %s", datetime.datetime.now())
    print("Hello")

