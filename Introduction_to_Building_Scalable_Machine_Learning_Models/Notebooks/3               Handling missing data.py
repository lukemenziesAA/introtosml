# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Handling Missing Data*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = "dbfs:/mnt/azureml/credit_score_data/train.csv"

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .format("csv")
    .load(readPath)
)

label = "Credit_Score"

# COMMAND ----------

display(df)

# COMMAND ----------

df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for cleaning up the data

# COMMAND ----------

from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import col, when, percentile_approx
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning the data
# MAGIC This next steps cleans the data to ensure you get the best out of the data you're putting into the model. It is well known in data science that a model is only as good as the data you put in. Therefore, a lot of time should be taken in cleaning and preparing the data, making sure it's ready for training.

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are calculating the upper and lower percintile values. These will be used in the cell below it. 

# COMMAND ----------

upper_perc = df.select(percentile_approx("Num_of_Loan", 0.99)).collect()[0][0]
lower_perc = df.select(percentile_approx("Num_of_Loan", 0.05)).collect()[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are cleaning the data. Certain tasks require converting from a string to a float or integer. Outliers are also removed certain columns such as age. These are replaced with null values since the actual value is unknown. It is very likely to be an error with these values so they may need to be imputed or removed. This also applies for the number of loans. 

# COMMAND ----------

df = (
    df.withColumn("Age", col("Age").cast(IntegerType()))
    .withColumn("Annual_Income", col("Annual_Income").cast(FloatType()))
    .withColumn("Num_of_Loan", col("Num_of_Loan").cast(IntegerType()))
    .withColumn(
        "Num_of_Delayed_Payment", col("Num_of_Delayed_Payment").cast(IntegerType())
    )
    .withColumn("Changed_Credit_Limit", col("Changed_Credit_Limit").cast(FloatType()))
    .withColumn("Outstanding_Debt", col("Outstanding_Debt").cast(FloatType()))
    .withColumn(
        "Amount_invested_monthly", col("Amount_invested_monthly").cast(FloatType())
    )
    .withColumn("Monthly_Balance", col("Monthly_Balance").cast(FloatType()))
    .sort(["Customer_ID", "ID", "Month"])
    .withColumn(
        "Age",
        when(col("Age") > 100, None).when(col("Age") < 10, None).otherwise(col("Age")),
    )
    .withColumn(
        "Num_of_Loan",
        when(col("Num_of_Loan") > upper_perc, upper_perc)
        .when(col("Num_of_Loan") < lower_perc, lower_perc)
        .otherwise(col("Num_of_Loan")),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## User defined function
# MAGIC This section covers using UDFs to clean the data further. The UDF is used to convert the credit history age column from a string into a float. Convert the years and months string into just years.

# COMMAND ----------

from pyspark.sql.functions import udf

# COMMAND ----------

def get_history_age(in_str):
    if in_str == "NA":
        total_years = None
    else:
        try:
            years, next_str = in_str.split(" Years")
            months = next_str.split(" Months")[0].split("and ")[1]
            total_years = float(years) + float(months) / 12
        except:
            total_years = None
        return total_years


history_age = udf(lambda z: get_history_age(z), FloatType())

# COMMAND ----------

df = df.withColumn("Credit_History_Age (Years)", history_age("Credit_History_Age"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset

# COMMAND ----------

seed = 42
trainDF, testDF = df.sample(fraction=0.1).randomSplit([0.7, 0.3], seed=42)

print(
    "We have %d training examples and %d test examples."
    % (trainDF.count(), testDF.count())
)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the routines

# COMMAND ----------

from pyspark.ml.feature import (
    VectorIndexer,
    VectorAssembler,
    StringIndexer,
    OneHotEncoder,
)
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import Imputer, StandardScaler

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the pipeline with an imputer
# MAGIC The cell below used the PySpark imputer to within a pipeline to handle missing data. 

# COMMAND ----------

imp_cols = [
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
]
cat_feats = [
    "Month",
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]
num_feats = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Credit_History_Age (Years)",
]

out_cats = [i + "_catv" for i in cat_feats]
f_cats = [i + "_cat" for i in out_cats]
vec_feats = num_feats + f_cats

imputer = Imputer(inputCols=imp_cols, outputCols=imp_cols)
imputer.setStrategy("median")
stringIndexer = StringIndexer(
    inputCols=cat_feats + ["Credit_Score"], outputCols=out_cats + ["Credit_Score_num"]
)

vectorAssembler = VectorAssembler(
    inputCols=vec_feats, outputCol="rawFeatures", handleInvalid="skip"
)
ohe = OneHotEncoder(inputCols=out_cats, outputCols=f_cats)

standardScaler = StandardScaler(inputCol="rawFeatures", outputCol="scaledFeatures")
model = LogisticRegression(
    featuresCol="rawFeatures", labelCol="Credit_Score_num"
)  # maxIter=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the pipelines
# MAGIC

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline_imp = Pipeline().setStages(
    [imputer, stringIndexer, ohe, vectorAssembler, model]
)
pipeline_no_imp = Pipeline().setStages([stringIndexer, ohe, vectorAssembler, model])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Generating predictions with and without imputation

# COMMAND ----------

predictionDF_imp = pipeline_imp.fit(trainDF).transform(testDF)
predictionDF_no_imp = pipeline_no_imp.fit(trainDF).transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC As is shown below, there is slightly worse improvement with imputation using the PySpark library. This can be due to RandomForestClassifier having its own method for handling missing data. Sometimes it is better to try without it if the model can handle missing entries. Another option is to use mode sophisticated imputation methods such as k-Nearest Neighbour Imputers. This isn't avaiable currently in PySpark ml. For this, one would use another library such as Scikit-learn. 

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluatorf1 = (
    MulticlassClassificationEvaluator()
    .setMetricName("fMeasureByLabel")
    .setMetricLabel(1)
    .setPredictionCol("prediction")
    .setLabelCol("Credit_Score_num")
)
evaluatorac = (
    MulticlassClassificationEvaluator()
    .setMetricName("accuracy")
    .setPredictionCol("prediction")
    .setLabelCol("Credit_Score_num")
)

f1 = evaluatorf1.evaluate(predictionDF_imp)
ac = evaluatorac.evaluate(predictionDF_imp)
print("Imputation F1 = %f" % f1)
print("Imputation Accuracu = %f" % ac)
f1 = evaluatorf1.evaluate(predictionDF_no_imp)
ac = evaluatorac.evaluate(predictionDF_no_imp)
print("No Imputation F1 = %f" % f1)
print("No Imputation Accuracu = %f" % ac)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the kNN imputer
# MAGIC Below show's the routine for using kNN imputer within Scikit-learn. There are other libraries that can be used but this one will be shown in this example. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert the to Pandas Dataframes
# MAGIC Here we are converting to Pandas Dataframes and creating training and test sets. 

# COMMAND ----------

train = trainDF.toPandas().set_index(["ID", "Customer_ID"])
test = testDF.toPandas().set_index(["ID", "Customer_ID"])

# COMMAND ----------

X_train = train.drop(label, axis=1)
y_train = train[label]
X_test = test.drop(label, axis=1)
y_test = test[label]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing the libraries 

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import roc_auc_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generating the pipelines
# MAGIC Here two pipelines are generated. One not using kNN within the RandomForestClassifer model and the other with. 

# COMMAND ----------

transformer_knn = ColumnTransformer([('categories', OneHotEncoder(), cat_feats), ('imputation', KNNImputer(), num_feats)])
transformer_simple = ColumnTransformer([('categories', OneHotEncoder(), cat_feats), ('imputation', SimpleImputer(), num_feats)])
transformer = ColumnTransformer([('categories', OneHotEncoder(), cat_feats)])
pipeline_simp_imp = Pipeline([('trans', transformer_simple), ('model', RandomForestClassifier())])
pipeline_knn_imp = Pipeline([('trans', transformer_knn), ('model', RandomForestClassifier())])
pipeline_no_imp = Pipeline([('trans', transformer), ('model', RandomForestClassifier())])

# COMMAND ----------

# MAGIC %md
# MAGIC Below shows a slights improvement in score with the kNN imputer. Althrough the imrovement is very small, training and test set are a much smaller size than the original dataset (for the purpose of this demo to speed up computational time). On the full dataset there is higher improvement. 

# COMMAND ----------

pipeline_no_imp.fit(X_train, y_train)
score = pipeline_no_imp.score(X_test, y_test)
print("Score with no imputation {:.2f}".format(score))

# COMMAND ----------

pipeline_simp_imp.fit(X_train, y_train)
score = pipeline_simp_imp.score(X_test, y_test)
print("Score with Simple imputation {:.2f}".format(score))

# COMMAND ----------

pipeline_knn_imp.fit(X_train, y_train)
score = pipeline_knn_imp.score(X_test, y_test)
print("Score with kNN imputation {:.2f}".format(score))
