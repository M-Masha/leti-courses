from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import countDistinct, col
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

df = spark.read.csv('/user/bigdata-course/Los Angeles Building and Safety Permits/building-and-safety-permit-information.csv', header = True, inferSchema = True, ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
df = df.select('Permit Sub-Type', 'Zone', 'Council District', 'Permit Type', 'Initiating Office', 'Permit Category')

values_to_keep = ["Electrical", "Bldg-Alter/Repair", "Plumbing", "HVAC", "Fire Sprinkler", "Bldg-Addition", "Bldg-New", "Grading", "Nonbldg-New", "Swimming-Pool/Spa", "Bldg-Demolition", "Sign", "Elevator", "Nonbldg-Alter/Repair", "Pressure Vessel", "Nonbldg-Addition", "Nonbldg-Demolition", "Bldg-Relocation"]
fdf = df.filter(col("Permit Type").isin(values_to_keep))
#fdf.groupBy('Permit Type').count().sort('count', ascending=False).show(30)

values_to_keep = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
fdf = fdf.filter(col("Council District").isin(values_to_keep))
#fdf.groupBy('Council District').count().sort('count', ascending=False).show(30)

values_to_keep = ["1 or 2 Family Dwelling", "Commercial", "Apartment", "Onsite", "Special Equipment", "Public Safety Only", "Offsite"]
fdf = fdf.filter(col("Permit Sub-Type").isin(values_to_keep))
#fdf.groupBy('Permit Sub-Type').count().sort('count', ascending=False).show(30)

values_to_keep = df.groupBy('Zone').count().sort('count', ascending=False).limit(100).select('Zone').rdd.flatMap(lambda x: x).collect()
fdf = fdf.filter(col("Zone").isin(values_to_keep))
#fdf.groupBy('Zone').count().sort('count', ascending=False).show(100)

values_to_keep = ["METRO", "VAN NUYS", "INTERNET", "WEST LA", "SOUTH LA", "SANPEDRO"]
fdf = fdf.filter(col("Initiating Office").isin(values_to_keep))
#fdf.groupBy('Initiating Office').count().sort('count', ascending=False).show(30)

values_to_keep = ["No Plan Check", "Plan Check"]
fdf = fdf.filter(col("Permit Category").isin(values_to_keep))
#fdf.groupBy('Permit Category').count().sort('count', ascending=False).show(30)

### AMOUNT OF DATA LEFT
#st = df.count()
#en = fdf.count()
#lft = en/st
#print(lft)
###

# Convert categorical features to numerical features
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="skip").fit(fdf) for column in ['Permit Sub-Type', 'Zone', 'Council District', 'Permit Type', 'Initiating Office']]
pipeline = Pipeline(stages=indexers)
fdf = pipeline.fit(fdf).transform(fdf)

# Alternative conversion for Districts
#fdf = fdf.withColumn('Council District Index', fdf['Council District'].cast('double'))

# Assemble all features into a single vector
assembler = VectorAssembler(inputCols=['Permit Sub-Type_index', 'Zone_index', 'Council District_index', 'Permit Type_index', 'Initiating Office_index'], outputCol='features')
fdf = assembler.transform(fdf)

# Convert Permit Type to numerical as 'label'
indexer = StringIndexer(inputCol = 'Permit Category', outputCol = 'label')
fdf = indexer.fit(fdf).transform(fdf)

# Leave the 50% of the dataset
fdf = fdf.sample(False, 0.5)

# Split the dataset
trainingData, testData = fdf.randomSplit([0.7, 0.3], seed = 101)

###############################################
start_time = time.time()

# Define the RFC
rf = RandomForestClassifier(labelCol='label', featuresCol='features', maxBins=100, maxDepth=15, numTrees=100, seed=101)

# Define the parameter grid
#paramGrid = (ParamGridBuilder().addGrid(rf.maxDepth, [5, 10, 15]).addGrid(rf.numTrees, [10, 50, 100]).build())

# Define the cross-validator
#cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=MulticlassClassificationEvaluator(), numFolds=5)

# Fit the model
#model = cv.fit(trainingData)
model = rf.fit(trainingData)

# Make predictions
predictions = model.transform(testData)

exec_time = time.time() - start_time
###############################################

# Select example rows to display
#result = predictions.select("prediction", "label", "features")
#result.show()
#result.select('prediction').distinct().show()

print(exec_time)

df = spark.read.csv('/user/bigdata-course/Los Angeles Building and Safety Permits/building-and-safety-permit-information.csv', header = True, inferSchema = True, ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
df = df.select('Permit Sub-Type', 'Zone', 'Council District', 'Permit Type', 'Initiating Office', 'Permit Category')

values_to_keep = ["Electrical", "Bldg-Alter/Repair", "Plumbing", "HVAC", "Fire Sprinkler", "Bldg-Addition", "Bldg-New", "Grading", "Nonbldg-New", "Swimming-Pool/Spa", "Bldg-Demolition", "Sign", "Elevator", "Nonbldg-Alter/Repair", "Pressure Vessel", "Nonbldg-Addition", "Nonbldg-Demolition", "Bldg-Relocation"]
#values_to_keep = ["Electrical", "Bldg-Alter/Repair", "Plumbing", "HVAC", "Fire Sprinkler", "Bldg-Addition"]
fdf = df.filter(col("Permit Type").isin(values_to_keep))
#fdf.groupBy('Permit Type').count().sort('count', ascending=False).show(30)

values_to_keep = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
fdf = fdf.filter(col("Council District").isin(values_to_keep))
#fdf.groupBy('Council District').count().sort('count', ascending=False).show(30)

values_to_keep = ["1 or 2 Family Dwelling", "Commercial", "Apartment", "Onsite", "Special Equipment", "Public Safety Only", "Offsite"]
fdf = fdf.filter(col("Permit Sub-Type").isin(values_to_keep))
#fdf.groupBy('Permit Sub-Type').count().sort('count', ascending=False).show(30)

values_to_keep = df.groupBy('Zone').count().sort('count', ascending=False).limit(100).select('Zone').rdd.flatMap(lambda x: x).collect()
fdf = fdf.filter(col("Zone").isin(values_to_keep))
#fdf.groupBy('Zone').count().sort('count', ascending=False).show(100)

values_to_keep = ["METRO", "VAN NUYS", "INTERNET", "WEST LA", "SOUTH LA", "SANPEDRO"]
fdf = fdf.filter(col("Initiating Office").isin(values_to_keep))
#fdf.groupBy('Initiating Office').count().sort('count', ascending=False).show(30)

values_to_keep = ["No Plan Check", "Plan Check"]
fdf = fdf.filter(col("Permit Category").isin(values_to_keep))
#fdf.groupBy('Permit Category').count().sort('count', ascending=False).show(30)

# Convert categorical features to numerical features
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid="keep").fit(fdf) for column in ['Permit Sub-Type', 'Zone', 'Council District', 'Permit Type', 'Initiating Office']]
pipeline = Pipeline(stages=indexers)
fdf = pipeline.fit(fdf).transform(fdf)

# Convert Permit Type to numerical as 'label'
indexer = StringIndexer(inputCol = 'Permit Category', outputCol = 'label')
fdf = indexer.fit(fdf).transform(fdf)

# Drop the original categorical columns
fdf = fdf.drop('Permit Sub-Type', 'Zone', 'Council District', 'Permit Type', 'Initiating Office', 'Permit Category')

# Leave the 50% of the dataset
fdf = fdf.sample(False, 0.5)

# Split the dataset
trainingData, testData = fdf.randomSplit([0.7, 0.3], seed = 101)
trainingData=trainingData.rdd
testData=testData.rdd

labeled_trainingData = trainingData.map(lambda row: LabeledPoint(row[-1], row[0:-1]))
labeled_testData = testData.map(lambda row: LabeledPoint(row[-1], row[0:-1]))

start_time = time.time()

# Train the rfc model
model = RandomForest.trainClassifier(labeled_trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=100, featureSubsetStrategy="all", impurity="gini", maxDepth=15, maxBins=100, seed=1234)

# Evaluate the model on the test set
predictions = model.predict(labeled_testData.map(lambda x: x.features))

exec_time = time.time() - start_time

print("Time:",exec_time)

from pyspark.sql.types import StructType, StructField, FloatType
print(predictions.take(20))
unique_predictions = predictions.distinct()
#print(unique_predictions.collect())
countp=predictions.count()
labelsRDD = labeled_testData.map(lambda lp: lp.label)
print(labelsRDD.take(20))
unique_labels = labelsRDD.distinct()
countl=labelsRDD.count()
print(unique_labels.collect())
print(countp,countl)

schema = StructType([StructField("prediction", FloatType(), True)])
df_pre = spark.createDataFrame(predictions.map(lambda x: (x,)), schema)
schemal = StructType([StructField("label", FloatType(), True)])
df_lab = spark.createDataFrame(labelsRDD.map(lambda x: (x,)), schemal)

from pyspark.sql.functions import monotonically_increasing_id

# Add a new column to both dataframes with unique values
df_pre = df_pre.withColumn("id", monotonically_increasing_id())
df_lab = df_lab.withColumn("id", monotonically_increasing_id())

# Join the two dataframes on the new column
df_joined = df_pre.join(df_lab, "id", "inner")

# Drop the new column from the resulting dataframe
df_joined = df_joined.drop("id")
num_rows = df_joined.count()

# Print the result
print("Number of rows: ", num_rows)

#Count the number of matching values
num_matches = df_joined.filter(col("prediction") == col("label")).count()

print("Number of matches: ", num_matches)
accuracy_percent = num_matches * 100 / float(num_rows)
print("accuracy: ", accuracy_percent)

evaluator = MulticlassClassificationEvaluator()
evaluator.setPredictionCol("prediction")
df_joined = df_joined.withColumn('prediction', df_joined['prediction'].cast('double'))

# for 0.0
print("Class %s precision = %s" % (0.0, evaluator.evaluate(df_joined, {evaluator.metricName: "precisionByLabel", evaluator.metricLabel: 0.0})))
print("Class %s recall = %s" % (0.0, evaluator.evaluate(df_joined, {evaluator.metricName: "recallByLabel", evaluator.metricLabel: 0.0})))
print("Class %s F1 score = %s" % (0.0, evaluator.evaluate(df_joined, {evaluator.metricName: "fMeasureByLabel", evaluator.metricLabel: 0.0})))

# for 1.0
print("Class %s precision = %s" % (1.0, evaluator.evaluate(df_joined, {evaluator.metricName: "precisionByLabel", evaluator.metricLabel: 1.0})))
print("Class %s recall = %s" % (1.0, evaluator.evaluate(df_joined, {evaluator.metricName: "recallByLabel", evaluator.metricLabel: 1.0})))
print("Class %s F1 score = %s" % (1.0, evaluator.evaluate(df_joined, {evaluator.metricName: "fMeasureByLabel", evaluator.metricLabel: 1.0})))

print("Accuracy = %s" % (evaluator.evaluate(df_joined, {evaluator.metricName: "accuracy"})))

# weighted
print("Weighted precision = %s" % (evaluator.evaluate(df_joined, {evaluator.metricName: "weightedPrecision"})))
print("Weighted recall = %s" % (evaluator.evaluate(df_joined, {evaluator.metricName: "weightedRecall"})))
print("Weighted F1 score = %s" % (evaluator.evaluate(df_joined, {evaluator.metricName: "weightedFMeasure"})))