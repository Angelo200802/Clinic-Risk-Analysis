from src.spark_manager import get_session, load_dataset
from src.model.logistic_reg import indexer_gender, indexer_risk, assembler, evaluate_model, fit
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
from dotenv import load_dotenv
import os

load_dotenv()
DS_PATH = os.getenv("DATASET_PATH")
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")

spark = get_session()   
ds = load_dataset(spark, DS_PATH)

rf_clf = RandomForestClassifier(labelCol="RiskCategory_b", featuresCol="features", seed=42)

pipe = Pipeline(stages=[indexer_gender, indexer_risk, assembler,rf_clf])

param_grifd = ( ParamGridBuilder()
             .addGrid(rf_clf.numTrees, [10,50,100])
             .addGrid(rf_clf.maxDepth, [10,15,20])
             .addGrid(rf_clf.impurity, ["gini", "entropy"])
             .build() 
             )

evaluator = BinaryClassificationEvaluator(labelCol="RiskCategory_b",metricName="areaUnderROC")
cv = CrossValidator(
    estimator=pipe,           
    estimatorParamMaps=param_grifd,
    evaluator=evaluator,
    seed=42,
    parallelism=2,
    numFolds=10                    
)

if __name__ == "__main__":
    train, test = ds.randomSplit([0.7, 0.3], seed=42)
    #fit(cv,train,test,"RiskCategory_b","Random Forest Classifier")
    model = PipelineModel.load("./src/model/saved_models/random_forest_pipeline_model")
    importances = model.stages[-1].featureImportances
    # Se l'ID è nella lista, vedrai qualcosa del genere:
    for i, column in enumerate(assembler.getInputCols()):
        print(f"Feature: {column:20} Importance: {importances[i]:.4f}")
    predictions = model.transform(test)
    evaluate_model(predictions,"RiskCategory_b")