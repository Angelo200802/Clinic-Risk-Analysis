from pyspark.ml.feature import StandardScaler
from src.spark_manager import get_session, load_dataset
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StringIndexer
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
    inputCols=[col for col in ds.columns if col not in  [ds.columns[-1],"Gender", "Timestamp", "Patient ID", "Height", "Weight", "Systolic Blood Pressure", "Diastolic Blood Pressure"]],
    outputCol="features_unscaled"
)

scaler = StandardScaler(
    inputCol="features_unscaled", 
    outputCol="features", 
    withStd=True, 
    withMean=True
)

lr = LogisticRegression(labelCol="RiskCategory_b", featuresCol="features")

pipe = Pipeline(stages=[indexer_gender, indexer_risk, assembler,scaler, lr])

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.0001,0.001,0.01,0.1,1])#[i for i in np.arange(0,0.11,0.01)])
             .addGrid(lr.elasticNetParam, [0.7,0.8,0.9])#[i for i in np.arange(0,1.1,0.1)]) 
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

def evaluate_model(predictions,label):
    evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction")
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def fit(cv:CrossValidator,train,save:bool = True,path = ""):
    model = cv.fit(train) 
    best_model = model.bestModel
    clf_model = best_model.stages[-1]
    #print(f"Summary:\n {clf_model.explainParams()}")
    if save:
        try:
            best_model.save(f"{path}_pipeline")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    return best_model

if __name__ == "__main__":
    train, test = ds.randomSplit([0.7, 0.3], seed=42)
    model = fit(cv,train,path=".src/model/saved_models/log_reg")
    #model = PipelineModel.load("./src/model/saved_models/log_reg_pipeline")
    predictions = model.transform(test)
    predictions.groupBy("RiskCategory_b", "prediction").count().show()
    evaluate_model(predictions,"RiskCategory_b")
    feature_names = [col for col in ds.columns if col not in ["RiskCategory_b", "Gender", "Timestamp", "Patient ID"]]
    coefficients = model.stages[-1].coefficients.toArray()
    for name, coef in zip(feature_names, coefficients):
        print(f"{name:25} : {coef:.4f}")
    print("Regularization Parameter:", model.stages[-1].regParam)
    print("ElasticNet Parameter:", model.stages[-1].elasticNetParam)
    print("Max Iterations:", model.stages[-1].maxIter)