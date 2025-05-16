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

# Install system-level dependencies
RUN apt-get update && apt-get install -y libgl1

# Create working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Create environment and install dependencies
RUN conda create -n appenv python=3.9 -y \
 && conda install -n appenv -c conda-forge \
    dlib \
    face-recognition \
    opencv \
    numpy \
    pandas \
    streamlit \
 && conda clean -a

# Set environment so container uses conda env by default
ENV PATH /opt/conda/envs/appenv/bin:$PATH
ENV CONDA_DEFAULT_ENV=appenv

# Expose Streamlit default port
EXPOSE 8501

# Start Streamlit (Render provides $PORT)
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false
