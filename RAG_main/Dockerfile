# Dockerfile

# --- Stage 1: Use an official Python runtime as a parent image ---
# Using python:3.10-slim is a good balance of features and size.
FROM python:3.10-slim

# --- Set the working directory inside the container ---
# All subsequent commands will run from this directory.
WORKDIR /app

# --- Copy the requirements file into the container ---
# This is done first to leverage Docker's layer caching.
# If requirements.txt doesn't change, Docker won't re-install dependencies on rebuilds.
COPY requirements.txt .

# --- Install the Python dependencies ---
# --no-cache-dir makes the image smaller.
# The sentence-transformers library will automatically download the embedding model
# when it's first used, and it will be cached within this layer.
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy the rest of your application code into the container ---
# This includes main.py and your .env file.
COPY . .

# --- Expose the port the app runs on ---
# This tells Docker that the container listens on port 8000.
# Render.com will automatically use this port.
EXPOSE 8000

# --- Define the command to run your application ---
# This is the command that will be executed when the container starts.
# It runs the Uvicorn server, making it accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]