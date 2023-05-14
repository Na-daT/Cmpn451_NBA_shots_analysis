import os
import sys
from pyspark.sql import SparkSession
#from pyspark import SparkContext, SparkConf, Spark

#os.environ['PYSPARK_PYTHON'] = sys.executable
#os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# create a SparkContext object
spark = SparkSession.builder.appName("NBA_Shot_Analysis").getOrCreate()
data = spark.read.csv("./NBA shot log 16-17-regular season/Shot data/*.csv", header=True, inferSchema=True)

print("Hi Count:", data.count())
print("Hi Take:", data.take(1))

# Data Pretty Printer
# data.show()

# stop the SparkContext object
spark.stop()



# The below commented code is for reference when using MapReduce with pyspark

# use filter to remove rows with null values in the "Name" or "Type" columns
#data = data.filter(lambda row: row != "" and row.split(",")[0] != "" and row.split(",")[1] != "")

# use map to convert the "Type" column to number encoding
#rdd = rdd.map(lambda row: (row.split(",")[0], int(row.split(",")[1].replace("Type", ""))))

# use reduceByKey to count the number of occurrences of each name for each type
#counts = rdd.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)

# print the name counts
#for (name, type), count in counts.collect():
#    print("{} (Type {}): {}".format(name, type, count))
