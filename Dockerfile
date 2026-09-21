FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

ENV INGEST_UPLOAD_DIR=/data/uploads INGEST_RESULT_DIR=/data/results
RUN mkdir -p /data/uploads /data/results

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
