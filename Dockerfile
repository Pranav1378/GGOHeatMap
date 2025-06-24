FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

USER root
RUN apt-get update && apt-get install -y python3-venv python3-pip unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 LANG=C.UTF-8 PYDECK_HEADLESS=1

COPY requirements.txt /tmp/
RUN pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

# only the Streamlit app is needed now
COPY app.py /app/app.py
WORKDIR /app

EXPOSE 8080
ENV PORT=8080
CMD ["streamlit","run","app.py","--server.port=8080","--server.address=0.0.0.0"]
