# Fase 1: Build
FROM debian:latest AS build-env

RUN apt-get update && apt-get install -y curl git wget unzip xz-utils zip libglu1-mesa
RUN useradd -ms /bin/bash flutteruser
USER flutteruser
WORKDIR /home/flutteruser

# Scarica l'SDK di Flutter
RUN git clone https://github.com/flutter/flutter.git -b stable

# Definiamo il PATH in modo che sia disponibile ovunque
ENV FLUTTER_HOME="/home/flutteruser/flutter"
ENV PATH="${FLUTTER_HOME}/bin:${FLUTTER_HOME}/bin/cache/dart-sdk/bin:${PATH}"

# Forziamo il download degli artefatti (questo rende i comandi successivi più veloci)
RUN flutter precache

WORKDIR /home/flutteruser/app

# COPIA: Assicurati che il percorso sia relativo a dove si trova il Dockerfile
# Se il Dockerfile è in "frontend/", e il progetto è in "frontend/frontend_clinic_risk/"
# il comando corretto è:
COPY --chown=flutteruser:flutteruser ./frontend/frontend_clinic_risk/pubspec.yaml .

# Eseguiamo pub get usando il percorso assoluto per sicurezza
RUN /home/flutteruser/flutter/bin/flutter pub get

# Copia il resto del codice
COPY --chown=flutteruser:flutteruser ./frontend/frontend_clinic_risk/ .
RUN rm -rf .dart_tool .packages .flutter-plugins .flutter-plugins-dependencies pubspec.lock

# Build finale
RUN /home/flutteruser/flutter/bin/flutter build web --release

# Fase 2: Nginx
FROM nginx:alpine
COPY --from=build-env /home/flutteruser/app/build/web /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]