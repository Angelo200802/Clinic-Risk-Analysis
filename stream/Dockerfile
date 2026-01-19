FROM python:3.9-slim

WORKDIR /appstream

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./stream .

CMD ["python", "./stream/main.py"]