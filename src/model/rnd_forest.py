from src.spark_manager import get_session, load_dataset
from src.model.logistic_reg import indexer_gender, scaler,indexer_risk, evaluate_model, fit
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Bucketizer, VectorAssembler
from dotenv import load_dotenv
import os, matplotlib.pyplot as plt, seaborn as sns, pandas as pd

load_dotenv()
DS_PATH = os.getenv("DATASET_PATH")
SAVE_MODEL_PATH = os.getenv("SAVE_MODEL_PATH")

spark = get_session()   
ds = load_dataset(spark, DS_PATH)

bmi_buckets = [-float("inf"), 18.5, 25.0, 30.0, float("inf")]   
bmi_bucketizer = Bucketizer(
    splits=bmi_buckets,
    inputCol="Derived_BMI",
    outputCol="BMI_binned"
)

features_to_use = [
    "Heart Rate", "Respiratory Rate", "Body Temperature", 
    "Oxygen Saturation", "Age", "Derived_HRV", 
    "Derived_Pulse_Pressure", "Derived_MAP", 
    "Gender_b"
]

assembler = VectorAssembler(
    inputCols=features_to_use,
    outputCol="features_unscaled"
)

rf_clf = RandomForestClassifier(
    labelCol="RiskCategory_b", 
    maxMemoryInMB=1024, 
    featuresCol="features", 
    seed=42)

pipe = Pipeline(stages=[indexer_gender, indexer_risk, assembler,scaler,rf_clf])

param_grifd = ( ParamGridBuilder()
             .addGrid(rf_clf.numTrees, [10,50,100])
             .addGrid(rf_clf.maxDepth, [5,10,15])
             .addGrid(rf_clf.impurity, ["gini", "entropy"])
             .build() 
             )

evaluator = BinaryClassificationEvaluator(labelCol="RiskCategory_b",metricName="areaUnderROC")
cv = CrossValidator(
    estimator=pipe,           
    estimatorParamMaps=param_grifd,
    evaluator=evaluator,
    seed=42,
    parallelism=3,
    numFolds=5                    
)

def analyze_correlation(df:pd.DataFrame):
    # Calcolo della matrice di correlazione (metodo di Pearson)
    corr_matrix = df.corr()

    # Visualizzazione con Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matrice di Correlazione delle Feature')
    plt.show()

if __name__ == "__main__":
    train, test = ds.randomSplit([0.7, 0.3], seed=42)
    #ds_pd = ds.toPandas().drop(columns=["Timestamp","Patient ID"])
    #ds_pd = pd.get_dummies(ds_pd, columns=["Gender","Risk Category"],dtype=int,drop_first=True)
    #analyze_correlation(ds_pd)
    model = fit(cv,train,test,path="./src/model/saved_models/rnd_forest")
    #model = PipelineModel.load("./src/model/saved_models/random_forest_pipeline_model")
    importances = model.stages[-1].featureImportances
    # Se l'ID è nella lista, vedrai qualcosa del genere:
    for i, column in enumerate(assembler.getInputCols()):
        print(f"Feature: {column:20} Importance: {importances[i]:.4f}")
    predictions = model.transform(test)
    evaluate_model(predictions,"RiskCategory_b")