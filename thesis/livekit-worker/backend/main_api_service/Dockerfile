# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages
# (e.g., for Pillow, psycopg2, etc.). Add any other build-time dependencies here.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app/

# Make port available to the world outside this container (defined by PORT env var)
# EXPOSE $PORT # This is illustrative; EXPOSE is usually a fixed number. Uvicorn will bind to PORT.

# Define environment variable for the port (will be set by docker-compose)
ENV PORT 8000

# Run main.py when the container launches
# Ensure your main.py uses the PORT environment variable for uvicorn host and port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
