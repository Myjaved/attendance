FROM continuumio/miniconda3

WORKDIR /app

# Create conda environment and install all necessary packages
RUN conda create -n appenv python=3.9 \
 && conda install -n appenv -c conda-forge \
    dlib \
    face-recognition \
    opencv \
    numpy \
    pandas \
    streamlit \
 && conda clean -a

# Activate conda environment
SHELL ["conda", "run", "-n", "appenv", "/bin/bash", "-c"]

# Copy your project files
COPY . .

# Expose the port streamlit will run on
EXPOSE 8000

# Run the app with Streamlit
CMD ["conda", "run", "-n", "appenv", "streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
