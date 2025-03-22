# Use NVIDIA's CUDA image with Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

