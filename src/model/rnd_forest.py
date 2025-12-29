from src.spark_manager import get_session, load_dataset
from src.model.logistic_reg import indexer_gender, indexer_risk, assembler, fit
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
from dotenv import load_dotenv
import os

load_dotenv()
ds_path = os.getenv("DATASET_PATH")

spark = get_session()   
ds = load_dataset(spark, ds_path)

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
    model = PipelineModel.load("random_forest_pipeline_model")
    print(f"Numero di alberi (numTrees): {model.stages[-1].getNumTrees}")
    print(f"Profondità massima (maxDepth): {model.stages[-1].getOrDefault('maxDepth')}")
    print(f"Criterio di impurità (impurity): {model.stages[-1].getOrDefault('impurity')}")
    print(f"Seme casuale (seed): {model.stages[-1].getOrDefault('seed')}")