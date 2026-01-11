from .logistic_reg import indexer_gender, indexer_risk, evaluate_model, fit
from spark_manager import load_dataset
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os, json
from dotenv import load_dotenv
# 1. Scaler specifico per Naive Bayes (porta i valori nel range [0, 1])
# NB: Il Naive Bayes multinomiale non accetta i valori negativi dello StandardScaler
minMaxScaler = MinMaxScaler(
    inputCol="features", 
    outputCol="minMaxFeatures"
)

# 2. Definizione del modello Naive Bayes
# smoothing: evita probabilità zero per categorie non viste nel training

assembler = VectorAssembler(
    inputCols=[
        "Heart Rate", "Respiratory Rate", "Body Temperature", 
        "Oxygen Saturation", "Age", "Derived_HRV", "Derived_BMI", 
        "Derived_Pulse_Pressure", "Derived_MAP", "Gender_b"],
    outputCol="features")

nb = NaiveBayes(
    labelCol="RiskCategory_b", 
    featuresCol="minMaxFeatures", 
    smoothing=1.0, 
    modelType="gaussian"
)

# 3. Pipeline completa (senza Poly, Naive Bayes non ne trae vantaggio)
pipeline_nb = Pipeline(stages=[
    indexer_gender, 
    indexer_risk,
    assembler, 
    minMaxScaler, 
    nb
])

paramGrid = (ParamGridBuilder()
             .addGrid(nb.smoothing, [0.0, 0.5, 1.0, 2.0, 5.0])
             .build())

evaluator = BinaryClassificationEvaluator(
    labelCol="RiskCategory_b", 
    rawPredictionCol="probability", 
    metricName="areaUnderROC"
)

# 4. Configuriamo il CrossValidator
cv_nb = CrossValidator(
    estimator=pipeline_nb, # La tua pipeline con MinMaxScaler
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5, 
    seed=42
)

if __name__ == "__main__":
    load_dotenv()
    DS_PATH = os.getenv("DATASET_PATH")
    SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH") 
    ds = load_dataset(DS_PATH)
    train, test = ds.randomSplit([0.7, 0.3], seed=42)
    #model = fit(cv_nb, train,path=SAVE_MODEL_PATH+"/nb")
    model = PipelineModel.load(SAVE_MODEL_PATH+"/nb_pipeline")
    predictions = model.transform(test)
    metrics = evaluate_model(predictions=predictions, label="RiskCategory_b")
    with open(os.path.join(SAVE_MODEL_PATH, "metrics.json"), "r") as f:
        met = json.load(f)
        met["naive_bayes"] = metrics
        
    f.close()
    with open(SAVE_MODEL_PATH+"/metrics.json", "w") as f:    
        json.dump(met, f, indent=4)
    f.close()
    #train.select("Heart Rate", "Respiratory Rate", "Body Temperature", "Oxygen Saturation", "Age", "Derived_HRV", "Derived_Pulse_Pressure", "Derived_MAP").describe().show()