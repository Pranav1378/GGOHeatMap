# --- Base image with GDAL/PROJ ---
    FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

    # --- System deps: venv + pip ---
    USER root
    RUN apt-get update \
     && apt-get install -y python3-venv python3-pip unzip \
     && rm -rf /var/lib/apt/lists/*
    
    # --- Python virtual-env ---
    RUN python3 -m venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH"
    ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
        LC_ALL=C.UTF-8 LANG=C.UTF-8 PYDECK_HEADLESS=1
    
    # --- Python deps ---
    COPY requirements.txt /tmp/
    RUN pip install --upgrade pip \
     && pip install --no-cache-dir -r /tmp/requirements.txt
    
    # --- Run training at build time ---
    COPY train.py /app/train.py
    WORKDIR /app
    RUN python train.py          # <— writes model.pkl, grid.csv, obs.csv
    
    # --- Copy lightweight Streamlit app ---
    COPY app.py /app/app.py
    
    EXPOSE 8080
    ENV PORT=8080
    ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
    ENV STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
    ENV STREAMLIT_SERVER_ENABLE_CORS=false
    ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
    CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.timeout=300"]
    