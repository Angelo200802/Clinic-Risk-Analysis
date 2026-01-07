from pyspark.sql.types import StructType, StructField, DoubleType, StringType

schema = StructType([
    StructField("Heart_Rate", DoubleType(), True),
    StructField("Systolic_BP", DoubleType(), True),
    StructField("Diastolic_BP", DoubleType(), True),
    StructField("SpO2", DoubleType(), True),
    StructField("Body_Temperature", DoubleType(), True),
    StructField("Respiratory_Rate", DoubleType(), True)
])