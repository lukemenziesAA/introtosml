# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Introduction to Clustering*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = "dbfs:/mnt/azureml/credit_card_segmentation.csv"

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .format("csv")
    .load(readPath)
)

# COMMAND ----------

display(df.drop("CUST_ID", "CASH_ADVANCE_FREQUENCY"))

# COMMAND ----------

df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the vector assembler and the model
# MAGIC Below shows the vector assember and the linear regression model being generated. The vector assembler creates a vectorized column that contains all the features. This needs to done to before passing the dataframe to the regression model. Otherwise and error will be raised. The LinearRegression model takes the featuresCol as an argument (the vectorized column containing all the features), as well as which column is the label column. This differs from the Scikit-learn library convention where the label data is passed in a separate argument. 

# COMMAND ----------

feats = df.drop("CUST_ID").columns
vectorAssembler = VectorAssembler(
    inputCols=feats, outputCol="rawFeatures", handleInvalid="skip"
)


model = KMeans(featuresCol="rawFeatures", k=7)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a pipeline
# MAGIC The cell below is used to setup the pipeline for a machine learning model. A pipeline allows the user to orchastrate each step for a model including preparation, transformations and training. They are also a good tool to prevent data leakage that can happen in some transformation steps if not done correctly. 

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages([vectorAssembler, model])

# COMMAND ----------

outmod = pipeline.fit(df)
predictions = outmod.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluating the model
# MAGIC The cell below shows how the use can evaluate the model. Unlike regression and clustering, unsupervised methods don't have an explicit way of scoring models. Instead they can be evaluated looking at things like euclidean distance between clusters and visually examining the clustering. 

# COMMAND ----------

evaluator = ClusteringEvaluator(featuresCol="rawFeatures")

silhouette = evaluator.evaluate(predictions.select("rawFeatures", "prediction"))
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = outmod.stages[1].clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
