# Flask Framework Guide

## What is Flask?

Flask is a lightweight web framework for Python that makes it easy to build web applications. Think of it as a toolkit that helps you create websites and web services without having to write everything from scratch.

## Why Use Flask?

- **Simple to Learn**: Flask has a minimal setup and is beginner-friendly
- **Flexible**: You can build anything from simple websites to complex web applications
- **Lightweight**: Doesn't come with unnecessary features, keeping your app fast
- **Well-Documented**: Extensive documentation and community support

## Key Concepts

### Routes
Routes are URL patterns that tell Flask what to do when someone visits a specific web address.

```python
@app.route('/')
def home():
    return "Hello, World!"
```

### Templates
Templates are HTML files that can display dynamic content. Flask uses Jinja2 templating engine.

### Request Handling
Flask can handle different HTTP methods like GET (retrieving data) and POST (sending data).

## Jinja2 Templating Engine

### What is Jinja2?

Jinja2 is Flask's default templating engine that allows you to create dynamic HTML pages. Think of it as a way to mix Python code with HTML to create web pages that change based on data.

### Key Features

- **Variable Insertion**: Display Python variables in HTML using `{{ variable_name }}`
- **Control Structures**: Use loops and conditions with `{% for %}` and `{% if %}`
- **Template Inheritance**: Create base templates that other templates can extend
- **Filters**: Modify variables before displaying them (e.g., `{{ name|upper }}`)

### Basic Syntax

```html
<!-- Variables -->
<h1>Hello, {{ username }}!</h1>

<!-- Loops -->
{% for item in items %}
    <li>{{ item }}</li>
{% endfor %}

<!-- Conditionals -->
{% if user.is_authenticated %}
    <p>Welcome back!</p>
{% else %}
    <p>Please log in</p>
{% endif %}
```

### Template Inheritance

Create a base template (`base.html`):
```html
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}{% endblock %}</title>
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>
```

Extend it in other templates:
```html
{% extends "base.html" %}
{% block title %}Home Page{% endblock %}
{% block content %}
    <h1>Welcome to my site!</h1>
{% endblock %}
```

## Basic Flask App Structure

```
flask-app/
├── app.py          # Main application file
├── templates/      # HTML templates
├── static/         # CSS, JS, images
└── requirements.txt # Dependencies
```

## Getting Started

1. Install Flask: `uv add flask`
2. Create a simple app in `app.py`
3. Run with: `uv run app.py`
## WSGI (Web Server Gateway Interface)

### What is WSGI?

WSGI is a specification that defines how web servers communicate with web applications in Python. It acts as a bridge between your Flask application and the web server, ensuring they can work together regardless of which server you choose.

### Why WSGI Matters

- **Standardization**: Provides a common interface between web servers and Python web applications
- **Portability**: Your Flask app can run on any WSGI-compatible server
- **Scalability**: Enables deployment on production servers like Gunicorn, uWSGI, or mod_wsgi

### WSGI Flow Diagram

```
Client Request → Web Server → WSGI Server → Flask App → Response
    ↑                                                      ↓
    └──────────────── Response Path ←←←←←←←←←←←←←←←←←←←←←←←←←┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Browser   │───▶│ Web Server  │───▶│WSGI Gateway │───▶│ Flask App   │
│             │    │ (Nginx/     │    │ (Gunicorn/  │    │             │
│             │    │ Apache)     │    │ uWSGI)      │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### WSGI in Practice

Flask has a built-in WSGI server for development:
```python
if __name__ == '__main__':
    app.run()  # Uses built-in WSGI server
```

For production, you'd use a dedicated WSGI server:
```bash
# Using Gunicorn
gunicorn -w 4 app:app

# Using uWSGI
uwsgi --http :8000 --wsgi-file app.py --callable app
```

### Common WSGI Servers

- **Gunicorn**: Popular, easy to configure
- **uWSGI**: Feature-rich, high performance
- **mod_wsgi**: For Apache integration
- **Waitress**: Pure Python, cross-platform