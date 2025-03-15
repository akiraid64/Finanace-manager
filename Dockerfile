# ====================================================================
# FINANCE MANAGER DOCKER SETUP
# ====================================================================
# This file contains instructions to build the Finance Manager app
# You don't need to understand this file to use the app
# Just run "docker-compose up" in the terminal and the app will start!
# ====================================================================

# Start with a system that has Python and the tools we need
FROM python:3.11-bullseye

# Create a folder for our app
WORKDIR /app

# Install all the C++ tools needed for the fast calculations
# This is what makes our financial calculations super fast!
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Get the list of Python packages we need
COPY requirements.txt .

# Install all the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all our app files to the Docker container
COPY . .

# Build the super-fast C++ calculation engine
RUN python build_cpp.py

# Make sure we have a place to store your financial data
RUN mkdir -p data

# Tell Docker which port our app will use
EXPOSE 5000

# Set up some technical settings for the app
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Start the web server when the container runs
# This makes the app available at http://localhost:5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--reuse-port", "--reload", "main:app"]

# ====================================================================
# HOW TO USE:
# 1. Install Docker on your computer
# 2. Open a terminal/command prompt in this folder
# 3. Run: docker-compose up
# 4. Go to http://localhost:5000 in your web browser
# ====================================================================