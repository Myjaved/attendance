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
# CMD ["bash", "-c", "conda run -n appenv streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false"]






FROM continuumio/miniconda3

# Install system-level dependencies (includes ffmpeg + OpenGL for cv2 imshow)
RUN apt-get update && \
    apt-get install -y libgl1 ffmpeg && \
    apt-get clean

WORKDIR /app

# Create conda environment and install dependencies
RUN conda create -n appenv python=3.9 \
 && conda install -n appenv -c conda-forge \
    dlib=19.24.2 \
    face-recognition=1.3.0 \
    opencv=4.5.5 \
    numpy=1.24.4 \
    pandas=1.5.3 \
    pillow=10.3.0 \
    streamlit=1.45.0 \
    imageio \
    imageio-ffmpeg \
 && conda clean -a

# Use conda environment for all following RUN/CMD lines
SHELL ["conda", "run", "-n", "appenv", "/bin/bash", "-c"]

COPY . .

# Optional for local testing (Render overrides it with $PORT)
EXPOSE 8501

# ✅ Entry point — make sure it uses $PORT
CMD ["bash", "-c", "conda run -n appenv streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false"]

