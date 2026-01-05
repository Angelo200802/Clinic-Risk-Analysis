from pyspark.ml.feature import StandardScaler
from src.spark_manager import get_session, load_dataset
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StringIndexer, PolynomialExpansion
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel
from dotenv import load_dotenv
import os, logging, numpy as np

load_dotenv()
DS_PATH = os.getenv("DATASET_PATH")
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")

spark = get_session()   
ds = load_dataset(spark, DS_PATH)

polyExpansion = PolynomialExpansion(
    degree=2, 
    inputCol="features", 
    outputCol="polyFeatures"
)

indexer_gender = StringIndexer(
    inputCol = "Gender",
    outputCol = "Gender_b",
    stringOrderType = "alphabetAsc"
)

indexer_risk = StringIndexer(
    inputCol = ds.columns[-1],
    outputCol = "RiskCategory_b",
    stringOrderType = "alphabetAsc"
)

assembler = VectorAssembler(
    inputCols=[col for col in ds.columns if col not in  [ds.columns[-1],"Gender", "Gender_b", "Timestamp", "Patient ID", "Weight (kg)", "Height (m)", "Systolic Blood Pressure", "Diastolic Blood Pressure"]],
    outputCol="features_unscaled"
)

assembler2 = VectorAssembler(
    inputCols=[col for col in ds.columns if col not in  [ds.columns[-1],"Gender", "Timestamp", "Patient ID", "Weight (kg)", "Height (m)", "Systolic Blood Pressure", "Diastolic Blood Pressure"]],
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features_unscaled", 
    outputCol="features1", 
    withStd=True, 
    withMean=True
)

lr = LogisticRegression(labelCol="RiskCategory_b", featuresCol="polyFeatures")

pipe = Pipeline(stages=[indexer_gender, indexer_risk, assembler,scaler,assembler2,polyExpansion, lr])

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.0001,0.001,0.01,0.1,1])#[i for i in np.arange(0,0.11,0.01)])
             .addGrid(lr.elasticNetParam, [0.0,0.5,0.9,1])#[i for i in np.arange(0,1.1,0.1)]) 
             .addGrid(lr.maxIter, [10,100,1000])#[i for i in range(0,100,10)])               
             .build())

evaluator = BinaryClassificationEvaluator(labelCol="RiskCategory_b",metricName="areaUnderROC")
cv = CrossValidator(
    estimator=pipe,           # Può essere il singolo modello o l'intera Pipeline
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    parallelism=4,
    numFolds=5                    # Divide i dati in 5 parti (5-fold cross validation)
)

def get_poly_feature_names(base_features, degree=2):
    """
    Ricostruisce i nomi delle feature generati da PolynomialExpansion (grado 2)
    L'ordine di Spark è: feature originali, poi le interazioni/quadrati
    """
    poly_names = []
    # 1. Feature originali
    poly_names.extend(base_features)
    
    # 2. Interazioni e quadrati (per grado 2)
    if degree >= 2:
        for i in range(len(base_features)):
            for j in range(i, len(base_features)):
                poly_names.append(f"{base_features[i]} * {base_features[j]}")
    
    return poly_names

def evaluate_model(predictions,label):
    evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction")
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)
    evaluator = BinaryClassificationEvaluator(labelCol="RiskCategory_b",metricName="areaUnderROC")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC ROC: {evaluator.evaluate(predictions)}")

def fit(cv:CrossValidator,train,save:bool = True,path = ""):
    model = cv.fit(train) 
    best_model = model.bestModel
    if save:
        try:
            best_model.save(f"{path}_pipeline")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    return best_model

if __name__ == "__main__":
    train, test = ds.randomSplit([0.7, 0.3], seed=42)
    model = fit(cv,train,path="./src/model/saved_models/log_reg2")
    #model = PipelineModel.load("./src/model/saved_models/log_reg_pipeline")
    
    predictions = model.transform(test)
    predictions.groupBy("RiskCategory_b", "prediction").count().show()
    evaluate_model(predictions,"RiskCategory_b")
    feature_names = [col for col in ds.columns if col not in [ds.columns[-1],"Gender", "Timestamp", "Patient ID", "Height (m)", "Weight (kg)", "Systolic Blood Pressure", "Diastolic Blood Pressure"]]
    coefficients = model.stages[-1].coefficients.toArray()
    print(f"Intercetta: {model.stages[-1].intercept:.4f}")
    print("\n--- Coefficients ---")
    # Genera i nomi mappati
    mapped_names = get_poly_feature_names(feature_names, degree=2)
    coeff_list = []
    for name, coeff in zip(mapped_names, coefficients):
        coeff_list.append((name, coeff))
    sorted_coeffs = sorted(coeff_list, key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':<40} | {'Coefficiente':<12}")
    print("-" * 55)
    for name, val in sorted_coeffs[:15]: # Vediamo i primi 15
        print(f"{name:<40} | {val:>12.4f}")
    print("Regularization Parameter:", model.stages[-1].getOrDefault('regParam'))
    print("ElasticNet Parameter:", model.stages[-1].getOrDefault('elasticNetParam'))
    print("Max Iterations:", model.stages[-1].getOrDefault('maxIter'))
    