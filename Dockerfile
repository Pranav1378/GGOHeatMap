# --- Base image with GDAL/PROJ ------------------------------------------------
    FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

    USER root
    RUN apt-get update \
     && apt-get install -y python3-venv python3-pip unzip git \
     && rm -rf /var/lib/apt/lists/*
    
    # --- Python venv --------------------------------------------------------------
    RUN python3 -m venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH" \
        PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
        PYDECK_HEADLESS=1 LC_ALL=C.UTF-8 LANG=C.UTF-8
    
    # --- install deps -------------------------------------------------------------
    COPY requirements.txt /tmp/req.txt
    RUN pip install --upgrade pip \
     && pip install --no-cache-dir -r /tmp/req.txt
    
    # --- run training at build time ----------------------------------------------
    COPY train.py /app/train.py
    WORKDIR /app
    RUN python train.py          # writes model.pkl, grid.csv, obs.csv
    
    # --- copy tiny UI app ---------------------------------------------------------
    COPY app.py /app/app.py
    
    EXPOSE 8080
    ENV PORT=8080
    CMD ["streamlit","run","app.py","--server.port=8080","--server.address=0.0.0.0"]
    