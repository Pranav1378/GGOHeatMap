# -------- base image with GDAL/PROJ pre-installed ---------------------------
    FROM osgeo/gdal:ubuntu-small-latest

    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        LC_ALL=C.UTF-8 \
        LANG=C.UTF-8
    
    # Streamlit needs this for headless mode
    ENV PYDECK_HEADLESS=1
    
    # Install Python deps
    COPY requirements.txt /tmp/req.txt
    RUN pip install --no-cache-dir -r /tmp/req.txt
    
    # Copy app
    COPY app.py /app/app.py
    WORKDIR /app
    
    EXPOSE 8080
    # Cloud Run looks for $PORT
    ENV PORT 8080
    CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
    