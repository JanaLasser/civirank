# Use an official NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install wget and bzip2, necessary for Miniconda installation
RUN apt-get update && apt-get install -y wget bzip2

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /miniconda && \
    rm miniconda.sh

# Add Miniconda to PATH
ENV PATH="/miniconda/bin:${PATH}"

# Copy the environment.yml file
COPY environment.yml /app/environment.yml

# Create the Conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ranker", "/bin/bash", "-c"]

# Ensure the Python executable used is from the conda environment
ENV PATH /miniconda/envs/ranker/bin:$PATH

# Copy the rest of your application's code
COPY . /app

# Expose the port the app runs on
EXPOSE 8000