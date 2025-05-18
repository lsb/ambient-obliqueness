# My Django App

## Overview
This is a basic Django application that includes user authentication, an SQLite database backend, and background processing capabilities.

## Project Structure
```
monolith
├── manage.py
├── myproject
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── app
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── migrations
│   │   └── __init__.py
│   ├── tests.py
│   └── views.py
├── worker.py
├── requirements.txt
└── README.md
```

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run migrations: `python manage.py migrate`
3. Start the server: `python manage.py runserver`

## Features
- User authentication using Django's built-in User model.
- AudioFrame model for storing audio data.
- Background processing with Celery.

## Usage
- Access the login page to authenticate users.
- Use the admin interface to manage AudioFrame entries.