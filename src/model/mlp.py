from .logistic_reg import indexer_gender, indexer_risk, evaluate_model, fit
from spark_manager import load_dataset
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
from dotenv import load_dotenv

# 1. Assembler delle feature base (9 input)
assembler = VectorAssembler(
    inputCols=[
        "Heart Rate", "Respiratory Rate", "Body Temperature", 
        "Oxygen Saturation", "Age", "Derived_HRV", "Derived_BMI",
        "Derived_Pulse_Pressure", "Derived_MAP"],
    outputCol="features")

# 2. Lo StandardScaler è obbligatorio per le Reti Neurali 
# per garantire che la discesa del gradiente converga correttamente.
scaler = StandardScaler(
    inputCol="features", 
    outputCol="scaledFeatures", 
    withStd=True, 
    withMean=True
)

assembler2 = VectorAssembler(
    inputCols=[
        "Heart Rate", "Respiratory Rate", "Body Temperature", 
        "Oxygen Saturation", "Age", "Derived_HRV", 
        "Derived_Pulse_Pressure", "Derived_MAP", "Gender_b"],
    outputCol="last_features")


ann = MultilayerPerceptronClassifier(
    labelCol="RiskCategory_b",
    layers= [10, 16, 8, 2],
    featuresCol="last_features",
    blockSize=128,
    seed=42,
    solver="l-bfgs" # Ottimizzatore basato sulla discesa del gradiente
)

# 4. Pipeline
pipeline_ann = Pipeline(stages=[
    indexer_gender, 
    indexer_risk,
    assembler, 
    scaler, 
    assembler2,
    ann
])

paramGrid = (ParamGridBuilder()
             .addGrid(ann.maxIter, [100,200,1000])
             .build())

evaluator = BinaryClassificationEvaluator(
    labelCol="RiskCategory_b", 
    rawPredictionCol="probability", 
    metricName="areaUnderROC"
)

# 6. CrossValidator
cv_ann = CrossValidator(
    estimator=pipeline_ann,
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
    
    # Split dei dati
    train, test = ds.randomSplit([0.7, 0.3], seed=42)
    
    print("Inizio addestramento ANN (Multilayer Perceptron)...")
    
    # Utilizzo della tua funzione fit per salvare il miglior modello
    #model_ann = fit(
    #    cv=cv_ann, 
    #    train=train, 
    #    save=True, 
    #    path=os.path.join(SAVE_MODEL_PATH, "ann_model")
    #)

    model_ann = PipelineModel.load(SAVE_MODEL_PATH+"/ann_model_pipeline")
    
    predictions = model_ann.transform(test) 
    predictions.groupBy("RiskCategory_b", "prediction").count().show()
    
    # Valutazione
    print("\n--- Risultati ANN ---")
    metrics = evaluate_model(predictions, "RiskCategory_b")
    import json
    with open(os.path.join(SAVE_MODEL_PATH, "metrics.json"), "r") as f:
        met = json.load(f)
        met["mlp"] = metrics
        
    f.close()
    with open(SAVE_MODEL_PATH+"/metrics.json", "w") as f:    
        json.dump(met, f, indent=4)
    f.close()
    # Stampa dei layer scelti dalla Cross Validation
    best_layers = model_ann.stages[-1].getLayers()
    print(f"\nConfigurazione Layers ottimale: {best_layers}")