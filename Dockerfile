# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files
# Ensure the requirements.txt file exists in the same directory as the Dockerfile
COPY ./requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Define environment variable
ENV NAME procressiveHI

# Run main.py when the container launches
CMD ["python", "main.py"]