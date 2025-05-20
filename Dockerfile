# FROM continuumio/miniconda3

# # Install system-level dependencies
# RUN apt-get update && apt-get install -y libgl1

# WORKDIR /app

# # Create conda environment and install dependencies
# RUN conda create -n appenv python=3.9 \
#  && conda install -n appenv -c conda-forge \
#     dlib \
#     face-recognition \
#     opencv \
#     numpy \
#     pandas \
#     streamlit \
#  && conda clean -a

# # Use conda environment for all following commands
# SHELL ["conda", "run", "-n", "appenv", "/bin/bash", "-c"]

# COPY . .

# EXPOSE 8501

# CMD ["conda", "run", "-n", "appenv", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.enableCORS=false"]



# FROM continuumio/miniconda3

# # Install system-level dependencies
# RUN apt-get update && apt-get install -y libgl1

# WORKDIR /app

# # Create conda environment and install dependencies
# RUN conda create -n appenv python=3.9 \
#  && conda install -n appenv -c conda-forge \
#     dlib \
#     face-recognition \
#     opencv \
#     numpy \
#     pandas \
#     streamlit \
#  && conda clean -a

# # Use conda environment for all following commands
# SHELL ["conda", "run", "-n", "appenv", "/bin/bash", "-c"]

# COPY . .

# # Expose default Streamlit port (for local dev) – optional
# EXPOSE 8501

# # ✅ Use the PORT environment variable provided by Railway
# CMD ["bash", "-c", "conda run -n appenv streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]



# FROM python:3.9-slim

# # System packages for dlib & OpenCV
# RUN apt-get update && apt-get install -y \
#     build-essential cmake \
#     libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy app files
# COPY . .

# EXPOSE 8501

# CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.enableCORS=false --server.enableXsrfProtection=false"]


FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 \
    build-essential cmake \
    libboost-all-dev \
    libopenblas-dev liblapack-dev libatlas-base-dev libx11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
