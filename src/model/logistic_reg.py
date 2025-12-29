from src.spark_manager import get_session, load_dataset
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from dotenv import load_dotenv
import os, logging, numpy as np

load_dotenv()
ds_path = os.getenv("DATASET_PATH")

spark = get_session()   
ds = load_dataset(spark, ds_path)

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
    inputCols=[col for col in ds.columns if col != ds.columns[-1] and col != "Gender" and col != "Timestamp" ],
    outputCol="features"
)

lr = LogisticRegression(labelCol="RiskCategory_b", featuresCol="features")
train, test = ds.randomSplit([0.7, 0.3], seed=42)

pipe = Pipeline(stages=[indexer_gender, indexer_risk, assembler, lr])

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.0001])  
             .addGrid(lr.elasticNetParam, [0.8])#[i for i in np.arange(0,1.1,0.1)]) 
             .addGrid(lr.maxIter, [1000,3000,100000])#[i for i in range(0,100,10)])               
             .build())

evaluator = BinaryClassificationEvaluator(labelCol="RiskCategory_b",metricName="areaUnderROC")
cv = CrossValidator(
    estimator=pipe,           # Può essere il singolo modello o l'intera Pipeline
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5                    # Divide i dati in 5 parti (5-fold cross validation)
)


def fit(cv:CrossValidator,train,test,label,name = ""):
    model = cv.fit(train)
    predictions = model.transform(test) 
    best_model = model.bestModel
    print(f"{name} Model Summary:")
    lr_model = best_model.stages[-1]
    training_summary = lr_model.summary
    print(f"Accuracy: {training_summary.accuracy}")
    print(f"Precision: {training_summary.weightedPrecision}")
    print(f"Recall: {training_summary.weightedRecall}")
    print(f"F1 Score: {training_summary.weightedFMeasure()}")   
    predictions.select("features", label, "prediction").show(10)
    try:
        best_model.save(f"{name}_pipeline_model")
        lr_model.save(f"{name}_model")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

    return best_model
if __name__ == "__main__":
    fit(cv,train,test,"RiskCategory_b","Logistic Regression")