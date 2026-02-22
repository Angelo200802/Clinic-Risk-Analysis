FROM python:3.9-slim-bullseye
# Installiamo OpenJDK 17 invece del 21
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    ca-certificates-java \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Aggiorna i percorsi per puntare a Java 17
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_SUBMIT_ARGS="--packages com.redislabs:spark-redis_2.12:2.4.2 --conf spark.redis.host=redis --conf spark.redis.port=6379 pyspark-shell"
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src .

CMD ["fastapi", "run", "./src/main.py", "--port", "8081", "--reload"]