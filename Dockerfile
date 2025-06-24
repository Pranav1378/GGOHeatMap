# 1. Base image (has GDAL/PROJ/GEOS)
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

# 2. Install python3-venv & pip so we can create venv
USER root
RUN apt-get update \
    && apt-get install -y python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 3. Create & activate a venv at /opt/venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 4. Python env vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYDECK_HEADLESS=1

# 5. Install Python deps into venv
COPY requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/req.txt

# 6. Copy app and set workdir
COPY app.py /app/app.py
WORKDIR /app

# 7. Expose and run
EXPOSE 8080
ENV PORT=8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
