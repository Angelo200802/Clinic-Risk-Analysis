# Human Vital Signs Anomaly Detection & Risk Classification 🏥

[cite_start]This repository contains a Big Data application developed with **Apache Spark** to automatically classify clinical risk levels from human physiological data[cite: 1, 2]. [cite_start]The system leverages **Machine Learning** and **Ensemble Learning** techniques to categorize health measurements into risk classes, such as low, medium, or high[cite: 3, 15].

## 📌 Project Overview
[cite_start]The application is designed to handle large-scale medical datasets and provides two primary processing workflows[cite: 1, 3]:
* [cite_start]**Batch Processing:** Analyzing historical data and performing complex clinical queries using **Spark SQL**[cite: 3, 9].
* [cite_start]**Real-time Streaming:** Monitoring continuous data flows (simulated or sensor-based) using **Spark Structured Streaming** to provide immediate risk alerts[cite: 3, 21].

## 📊 Dataset Features
[cite_start]The project utilizes the **Human Vital Sign Dataset** (approx. 200,000 measurements)[cite: 5], focusing on the following key physiological parameters:
* [cite_start]**Heart Rate** and **Respiratory Rate**[cite: 7].
* [cite_start]**Blood Pressure** (Systolic and Diastolic)[cite: 7].
* [cite_start]**Oxygen Saturation ($SpO_2$)** and **Body Temperature**[cite: 7].

## 🧠 Machine Learning Approach
[cite_start]To ensure high accuracy and robustness, the system implements an **Ensemble Learning** strategy[cite: 15, 19]. [cite_start]It combines several base learners through voting techniques[cite: 15, 19]:
* [cite_start]**Logistic Regression** and **Decision Tree Classifier**[cite: 16, 17].
* [cite_start]**Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)**[cite: 18, 19].

[cite_start]Performance is evaluated using standard metrics including **Accuracy, Precision, Recall, F1-score**, and **ROC-AUC**[cite: 19].

## 🛠 Technology Stack
* [cite_start]**Core Engine:** Apache Spark (SQL, MLlib, Structured Streaming)[cite: 2].
* [cite_start]**Data Analysis:** Spark SQL for descriptive and multivariate clinical queries[cite: 9, 10, 12].
* [cite_start]**Output:** Interactive dashboards featuring real-time statistics and model comparison[cite: 23].

---
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