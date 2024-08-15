# Use an official Python 3.11 base image for AMD64
FROM python:3.11-slim-bullseye

# Set the working directory
WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda for AMD64
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /miniconda && \
    rm miniconda.sh

# Install Miniconda for aarch64 (for container debugging on M1 Mac)
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh && \
#     chmod +x miniconda.sh && \
#     ./miniconda.sh -b -p /miniconda && \
#     rm miniconda.sh


# Add Miniconda to PATH
ENV PATH="/miniconda/bin:${PATH}"

# install mamba since conda has trouble properly resolving dependencies
RUN conda install -c conda-forge -y mamba

# Copy the environment.yml file
COPY environment.yml /app/environment.yml

# Create the Conda environment
RUN mamba env create --debug -f environment.yml
RUN mamba clean -afy

# Activate the conda environment
SHELL ["conda", "run", "-n", "ranker", "/bin/bash", "-c"]

# Install PyTorch for CPU (since we're not using CUDA in this setup)
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Install additional packages
RUN pip install --no-cache-dir \
    optimum[onnxruntime] \
    fastapi==0.111.0 \
    fasttext_wheel==0.9.2 \
    huggingface_hub==0.23.3 \
    lexicalrichness==0.5.1 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    pydantic==2.7.4 \
    ranking_challenge==2.0.0 \
    sentence_transformers==3.0.1 \
    transformers==4.41.2 \
    uvicorn==0.30.1 \
    simplejson \
    numexpr \
    bottleneck

# Copy the rest of your application's code
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Set the default command to run your FastAPI application
# CMD ["conda", "run", "--no-capture-output", "-n", "ranker", "python", "main.py", "--port", "8000", "--scroll_warning_limit", "-0.1", "--batch_size", "8"]
CMD ["python", "start_server.py", "--port", "8000", "--batch_size", "8", "--scroll_warning_limit", "-0.1"]
