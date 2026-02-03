FROM python:3.10-slim

WORKDIR /appstream

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./stream .


CMD ["python", "stream/stream_script.py"]