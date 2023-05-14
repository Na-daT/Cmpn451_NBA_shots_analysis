import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, mean, round
#from pyspark import SparkContext, SparkConf, Spark

#os.environ['PYSPARK_PYTHON'] = sys.executable
#os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# create a SparkContext object
spark = SparkSession.builder.appName("NBA_Shot_Analysis").getOrCreate()
data = spark.read.csv("./NBA shot log 16-17-regular season/Shot data/*.csv", header=True, inferSchema=True)
player_data = spark.read.csv("C:/Users/Muhab/Documents/GitHub/Cmpn451_NBA_shots_analysis/NBA shot log 16-17-regular season\Player Regular 16-17 Stats.csv", header=True, inferSchema=True)


player_data.show()
# Add a column to player_data that is the player's name combine the First Name and Last Name
player_data = player_data.withColumn("Name", player_data["#FirstName"] + " " + player_data["#LastName"])
# print player names
player_data.select("Name").show()

# remove the columns that are not needed [#Date/Time of Update: 2017-05-09 4:34:01 PM,
#   #Player ID, #Jersey Num, #Birth Date, #Birth City, #Birth Couuntry, #Team ID
#   #Team Abbr, #Team City, #Team Name]
player_data = player_data.drop("#Date/Time of Update: 2017-05-09 4:34:01 PM", "#Player ID", 
    "#Jersey Num", "#Birth Date", "#Birth City", "#Birth Country", 
    "#Team ID", "#Team Abbr", "#Team City", "#Team Name", "#FirstName", "#LastName", "#Position")

# join the two dataframes on the "Name" column of player_data with the shoot_player column of data
print("Number of rows in data 1: ", data.count())
data = data.join(player_data, data["shoot player"] == player_data["Name"], "inner")
print("Number of rows in data 2: ", data.count())
# remove the "Name", "shoot player" column from data
data = data.drop("shoot player", 'date', 'home game', 'away team', 'FtAtt', 'FtMade')
# drop players with 0 in Fg2PtAtt or Fg3PtAtt
data = data.filter(data["#Fg2PtAtt"] != 0 & data["#Fg3PtAtt"] != 0)
# if points = 2 then accuracy = Fg2PtMade / Fg2PtAtt, else acuracy = Fg3PtMade / Fg3PtAtt
data = data.withColumn("accuracy", when(data["points"] == 2, data["#Fg2PtMade"] / data["#Fg2PtAtt"]).otherwise(data["#Fg3PtMade"] / data["#Fg3PtAtt"]))

# print number of rows in data
print("Number of rows in data 3: ", data.count())

# Display count of nulls in #Age column
print("Count of nulls in #Age column: ", data.filter(data["#Age"].isNull()).count())

# fill nulls using round mean in #Age, #Weight, #Height
data.select(mean("#Age")).show() 
data = data.fillna({'#Age': round(data.select(mean("#Age"))), '#Weight': round(data.select(mean("#Weight"))),
                     '#Height': round(data.select(mean("#Height"))), 
                     "time from last shot": round(data.select(mean("time from last shot")))})

# print count of nulls in every column
for col in data.columns:
    print(col, "\t", "with null values: ", data.filter(data[col].isNull()).count())


# stop the SparkContext object
spark.stop()


# Data Pretty Printer
# data.show()


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
