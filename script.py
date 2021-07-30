from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from functools import reduce
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.mllib.evaluation import MulticlassMetrics

trainData = spark.read.format('csv').options(sep=';',skiprows=1,header=True, inferschema=True ).load('s3://aws-emr-resources-295537701162-us-east-2/TrainingDataset.csv')

testData = spark.read.format('csv').options(sep=';',skiprows=1,header=True, inferschema=True ).load('s3://aws-emr-resources-295537701162-us-east-2/ValidationDataset.csv')

oldColumns = testData.schema.names

newColumns = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']

testData = reduce(lambda testData, idx: testData.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), testData)

oldColumns = trainData.schema.names

trainData = reduce(lambda trainData, idx: trainData.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), trainData)

featureCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

assembler = VectorAssembler(inputCols=featureCols, outputCol="features") 

assembled_df = assembler.transform(trainData)

assembled_test_df = assembler.transform(testData)

standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

scaled_df = standardScaler.fit(assembled_df).transform(assembled_df)

scaled_test_df = standardScaler.fit(assembled_df).transform(assembled_test_df)

rf = RandomForestClassifier(labelCol="quality", featuresCol="features_scaled", numTrees=100)

pipeline = Pipeline(stages=[rf])

model = pipeline.fit(scaled_df)

predictions = model.transform(scaled_test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
evaluator2 = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="F1")


accuracy = evaluator.evaluate(predictions)
f1_score = evaluator2.evaluate(predictions)

print(F"Accuracy: {accuracy}")
print(F"F1 score: {f1_score }")
print("Test Error = %g" % (1.0 - accuracy))

preds_and_labels = predictions.select(['prediction','quality']).withColumn('quality', F.col('quality').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','quality'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())











