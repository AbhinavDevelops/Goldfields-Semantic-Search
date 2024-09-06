# Use the official Python image as the base image
FROM python:3

# Install system dependencies required for faiss, pandas, numpy, distutils, and setuptools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    python3-distutils \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/* 

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using Gunicorn
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8080" ]
