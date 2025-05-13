FROM continuumio/miniconda3

WORKDIR /app

# Create a conda environment and install packages
RUN conda create -n appenv python=3.9 \
 && conda install -n appenv -c conda-forge dlib face-recognition opencv numpy pandas \
 && conda clean -a

# Activate the conda environment
SHELL ["conda", "run", "-n", "appenv", "/bin/bash", "-c"]

COPY . .

EXPOSE 8000

CMD ["conda", "run", "-n", "appenv", "streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
