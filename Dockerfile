FROM continuumio/miniconda3

# Install system-level dependencies
RUN apt-get update && apt-get install -y libgl1

WORKDIR /app

# Create conda environment and install dependencies
RUN conda create -n appenv python=3.9 \
 && conda install -n appenv -c conda-forge \
    dlib \
    face-recognition \
    opencv \
    numpy \
    pandas \
    streamlit \
 && conda clean -a

# Use conda environment for all following commands
SHELL ["conda", "run", "-n", "appenv", "/bin/bash", "-c"]

COPY . .

EXPOSE 8501

CMD ["conda", "run", "-n", "appenv", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.enableCORS=false"]

