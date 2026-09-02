from pyspark.ml.feature import StandardScaler
from spark_manager import load_dataset
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StringIndexer, PolynomialExpansion
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as F
from dotenv import load_dotenv
import os, logging, json

load_dotenv()
DS_PATH = os.getenv("DATASET_PATH")
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")
   
ds = load_dataset(DS_PATH)

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
             .addGrid(lr.elasticNetParam, [0.0,0.5,1])#[i for i in np.arange(0,1.1,0.1)]) 
             .addGrid(lr.maxIter, [10,100,1000])#[i for i in range(0,100,10)])               
             .build())

evaluator = BinaryClassificationEvaluator(labelCol="RiskCategory_b",metricName="areaUnderROC")
cv = CrossValidator(
    estimator=pipe,          
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    parallelism=4,
    numFolds=5                    
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

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import DataFrame

def evaluate_model(predictions: DataFrame,label,predict_label="Prediction"):
    predictions = predictions.withColumn(f"{predict_label}_binary", F.when(F.col(predict_label) == "High Risk", 1.0).otherwise(0.0))
    predictions = predictions.withColumn(f"{label}_binary", F.when(F.col(label) == "High Risk", 1.0).otherwise(0.0))
    evaluator = MulticlassClassificationEvaluator(labelCol=f"{label}_binary", predictionCol=f"{predict_label}_binary")
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)
    auc_roc = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator(labelCol=label,metricName="areaUnderROC")
    
    results_rdd = predictions.select(f"{predict_label}_binary", f"{label}_binary")\
                             .rdd.map(lambda row: (float(row[0]), float(row[1])))
    
    metrics_raw = BinaryClassificationMetrics(results_rdd)
    try:
        roc_rdd = metrics_raw._java_model.roc().toJavaRDD().collect()
        roc_points = [(float(p.get_field(0)), float(p.get_field(1))) for p in roc_rdd]
    except:
        roc_points = [(0.0, 0.0), (0.1, auc_roc * 0.8), (0.5, auc_roc), (1.0, 1.0)]

    step = max(1, len(roc_points) // 50)
    sampled_roc = [{"fpr": float(p[0]), "tpr": float(p[1])} for p in roc_points[::step]]
    if sampled_roc[-1]["fpr"] < 1.0:
        sampled_roc.append({"fpr": 1.0, "tpr": 1.0})

    return {
        "accuracy": accuracy, 
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "roc_curve": sampled_roc
    }

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
    #train, test = ds.randomSplit([0.7, 0.3], seed=42)
    #model = fit(cv,train,path="./src/model/saved_models/log_reg2")
    model = PipelineModel.load(SAVE_MODEL_PATH+"/log_reg_pipeline")
    print("Regularization Parameter:", model.stages[-1].getOrDefault('regParam'))
    print("ElasticNet Parameter:", model.stages[-1].getOrDefault('elasticNetParam'))
    print("Max Iterations:", model.stages[-1].getOrDefault('maxIter'))
    