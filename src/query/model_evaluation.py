from fastapi import APIRouter, HTTPException
from spark_manager import load_dataset
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame, functions as F
import os, logging
from model.logistic_reg import evaluate_model
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()
router_model_ev = APIRouter()
ds: DataFrame = load_dataset(os.getenv("DATASET_PATH"))
ds.show(5)
def evaluate_by_category(df:DataFrame,category_col):
    categories = [ row[0] for row in df.select(category_col).distinct().collect() ]

    stratified_metrics = {}
    for category in categories:
        category_df = df.filter(F.col(category_col) == category)
        metrics = evaluate_model(predictions=category_df, label="Risk Category", predict_label="Prediction")
        metrics.pop("roc_curve", None)
        stratified_metrics[category] = metrics

    return stratified_metrics

def add_age_group(df:DataFrame):
    splits = [0.0, 18.0, 30.0, 50.0, 70.0, float("inf")]
    bucketizer = Bucketizer(splits=splits, inputCol="Age", outputCol="age_group_idx")
    
    df_with_idx = bucketizer.transform(df)
    return df_with_idx.withColumn(
        "Age_Range", 
        F.when(F.col("age_group_idx") == 0.0, "0-18")
         .when(F.col("age_group_idx") == 1.0, "18-30")
         .when(F.col("age_group_idx") == 2.0, "30-50")
         .when(F.col("age_group_idx") == 3.0, "50-70")
         .when(F.col("age_group_idx") == 4.0, "70+")
         .otherwise("Unknown")
    )

def add_bmi_category(df:DataFrame):
    splits = [0.0, 18.5, 25.0, 30.0, float("inf")]
    bucketizer = Bucketizer(splits=splits, inputCol="Derived_BMI", outputCol="bmi_category_idx")    
    
    df_with_idx = bucketizer.transform(df)
    return df_with_idx.withColumn(
        "BMI_Category", 
        F.when(F.col("bmi_category_idx") == 0.0, "Underweight")
         .when(F.col("bmi_category_idx") == 1.0, "Normal weight")
         .when(F.col("bmi_category_idx") == 2.0, "Overweight")
         .otherwise("Obese")
    )

df_evaluated_cat = {
    "Gender" : evaluate_by_category(ds, category_col="Gender"),
    "Age_Group" : evaluate_by_category(add_age_group(ds), category_col="Age_Range"),
    "BMI_Category" : evaluate_by_category(add_bmi_category(ds), category_col="BMI_Category")
}

evaluation = evaluate_model(predictions=ds,label="Risk Category", predict_label="Prediction")
evaluation_by_shock_risk = (
    evaluate_model(
        predictions = ds.withColumn(
            "ShockRisk",
            F.when(
                (F.col("Heart Rate") / F.col("Systolic Blood Pressure")) > 0.85, 1.0 
            ).otherwise(0.0)
        ),
        label="Risk Category",
        predict_label="ShockRisk" 
    )
)

ensemble_consensus= (
    ds.withColumn(
        "lr_hit", 
        F.when(
            ((F.col("Prediction") == "Low Risk") & (F.col("pred_logistic_regression") == 1.0)) |
            ((F.col("Prediction") == "High Risk") & (F.col("pred_logistic_regression") == 0.0)), 
            "LR"
        ).otherwise("")
    ).withColumn(
        "mlp_hit", 
        F.when(
            ((F.col("Prediction") == "Low Risk") & (F.col("pred_mlp") == 1.0)) |
            ((F.col("Prediction") == "High Risk") & (F.col("pred_mlp") == 0.0)), 
            "MLP"
        ).otherwise("")
    ).withColumn(
        "nb_hit", 
        F.when(
            ((F.col("Prediction") == "Low Risk") & (F.col("pred_naive_bayes") == 1.0)) |
            ((F.col("Prediction") == "High Risk") & (F.col("pred_naive_bayes") == 0.0)), 
            "NB"
        ).otherwise("")
    )
    .withColumn("combination", F.concat_ws("+", F.array_remove(F.array("lr_hit", "mlp_hit", "nb_hit"), "")))
    .withColumn("combination", F.when(F.col("combination") == "", "Tutti Sbagliano").otherwise(F.col("combination")))
    .groupBy("combination").count().orderBy(F.desc("count"))
).toPandas().to_dict(orient="records")

@router_model_ev.get("/evaluation/ensemble_consensus")
def get_ensemble_consensus():
    return {"data" : ensemble_consensus}

@router_model_ev.get("/evaluation/confusion_matrix")
def get_confusion_matrix():
    df_eval = ds.withColumn("Result_Type", 
    F.when((F.col("Prediction") == "High Risk") & (F.col("Risk Category") == "High Risk"), "TP")
     .when((F.col("Prediction") == "Low Risk") & (F.col("Risk Category") == "Low Risk"), "TN")
     .when((F.col("Prediction") == "High Risk") & (F.col("Risk Category") == "Low Risk"), "FP")
     .otherwise("FN")
    )
    
    confusion_matrix = df_eval.groupBy("Result_Type").count().collect()
    confusion_matrix = {row["Result_Type"]: row["count"] for row in confusion_matrix}
    return {
        "confusion_matrix": confusion_matrix
    }

@router_model_ev.get("/evaluation/metrics_shock_risk")
def get_metrics_shock_risk():
    if not evaluation_by_shock_risk :
        logging.error(f"Error during shock risk model evaluation")
        raise HTTPException(status_code=500, detail="Error during shock risk model evaluation")
    return evaluation_by_shock_risk

@router_model_ev.get("/evaluation/metrics")
def get_metrics():
    if not evaluation :
        logging.error(f"Error during model evaluation")
        raise HTTPException(status_code=500, detail="Error during model evaluation")
    return evaluation

@router_model_ev.get("/evaluation/evaluate_by_category")
def get_evaluation_by_category():
    if not df_evaluated_cat:
        logging.error(f"Error during category evaluation")
        raise HTTPException(status_code=500, detail="Error during category evaluation")
    return df_evaluated_cat
