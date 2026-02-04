# Clinic-Risk-Analysis

## App Architecture

![Architecture](img/architecture.png)

## How to Start

### Environment Variables

Before starting the application, set the following environment variables:

**App Service:**
- `DATASET_PATH`: Path to the dataset CSV file (default: `/app/src/data/human_vital_signs_dataset_2024.csv`)
- `SAVE_MODEL_PATH`: Path to save/load trained models (default: `/app/src/model/saved_models`)
- `REDIS_HOST`: Redis server host (default: `redis`)
- `REDIS_PORT`: Redis server port (default: `6379`)

**Stream Service:**
- `STREAM_GET`: URL endpoint for GET requests to the app service
- `STREAM_POST`: URL endpoint for POST requests to the app service
- `GEMINI_API_KEY`: API key for Google Gemini
- `GEMINI_API_MODEL`: Gemini model name to use

### Running the Application

To run the application, use Docker Compose:

```bash
docker-compose up --build 
```