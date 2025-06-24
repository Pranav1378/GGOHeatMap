# Base image with GDAL/PROJ pre-installed
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

# Install pip so we can install Python packages
USER root
RUN apt-get update \
    && apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Streamlit needs headless mode
ENV PYDECK_HEADLESS=1

# Install Python dependencies
COPY requirements.txt /tmp/req.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/req.txt

# Copy the Streamlit app
COPY app.py /app/app.py
WORKDIR /app

# Expose port and set CMD
EXPOSE 8080
ENV PORT 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
