import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, mean, round, concat_ws, col, split, expr, regexp_extract
from pyspark.sql.types import IntegerType
#from pyspark import SparkContext, SparkConf, Spark

#os.environ['PYSPARK_PYTHON'] = sys.executable
#os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# create a SparkContext object
spark = SparkSession.builder.appName("NBA_Shot_Analysis").getOrCreate()
data = spark.read.csv("./NBA shot log 16-17-regular season/Shot data/*.csv", header=True, inferSchema=True)
player_data = spark.read.csv("C:/Users/Muhab/Documents/GitHub/Cmpn451_NBA_shots_analysis/NBA shot log 16-17-regular season\Player Regular 16-17 Stats.csv", header=True, inferSchema=True)


# Add a column to player_data that is the player's name combine the First Name and Last Name
player_data = player_data.withColumn("Name", concat_ws(" ","#FirstName",'#LastName'))

# remove the columns that are not needed [#Date/Time of Update: 2017-05-09 4:34:01 PM,
#   #Player ID, #Jersey Num, #Birth Date, #Birth City, #Birth Couuntry, #Team ID
#   #Team Abbr, #Team City, #Team Name]
player_data = player_data.drop("#Date/Time of Update: 2017-05-09 4:34:01 PM", "#Player ID", 
    "#Jersey Num", "#Birth Date", "#Birth City", "#Birth Country", 
    "#Team ID", "#Team Abbr", "#Team City", "#Team Name", "#FirstName", "#LastName", "#Position")

# join the two dataframes on the "Name" column of player_data with the shoot_player column of data
data = data.join(player_data, data["shoot player"] == player_data["Name"], "inner")
# remove the "Name", "shoot player" column from data
data = data.drop("shoot player", 'date', 'home game', 'away team', 'FtAtt', 'FtMade')
# drop nulls in location x
data = data.dropna(subset=['location x', 'self previous shot', 'opponent previous shot', 'time from last shot'])
# drop players with 0 in Fg2PtAtt or Fg3PtAtt
data = data.filter((data["#Fg2PtAtt"] != 0) & (data["#Fg3PtAtt"] != 0))


# if points = 2 then accuracy = Fg2PtMade / Fg2PtAtt, else acuracy = Fg3PtMade / Fg3PtAtt
data = data.withColumn("accuracy", when(data["points"] == 2, data["#Fg2PtMade"] / data["#Fg2PtAtt"]).otherwise(data["#Fg3PtMade"] / data["#Fg3PtAtt"]))

# height mapping
height_map = {'5\'4\"\"': '64', '5\'9\"\"': '69', '5\'10\"\"': '70', '5\'11\"\"': '71',
       '6\'0\"\"': '72', '6\'1\"\"': '73', '6\'2\"\"': '74', '6\'3\"\"': '75', '6\'4\"\"': '76', '6\'5\"\"': '77', '6\'6\"\"': '78', '6\'7\"\"': '79', '6\'8\"\"': '80',
       '6\'9\"\"': '81', '6\'10\"\"': '82', '6\'11\"\"': '83', '7\'0\"\"': '84', '7\'1\"\"': '85', '7\'2\"\"': '86', '7\'3\"\"': '87'}
data = data.replace(height_map, subset=['#Height'])
data = data.withColumn("#Height", data["#Height"].cast(IntegerType()))

# fill nulls using round mean in #Age, #Weight, #Height
mean_age = int(data.select(round(mean("#Age"))).collect()[0][0])
mean_weight = int(data.select(round(mean("#Weight"))).collect()[0][0])
mean_height = int(data.select(round(mean("#Height"))).collect()[0][0])
mean_time_from_last_shot = int(data.select(round(mean("time from last shot"))).collect()[0][0])

data = data.fillna({'#Age': mean_age, '#Weight': mean_weight,
                     '#Height': mean_height, 'time from last shot': mean_time_from_last_shot})

# Normalize location x and location y
# for location x, we'll split the full length of the court into one half-court
# and then normalize the values to be between 0 and 1
data = data.withColumn("location x", when(data["location x"] > 470, (940 - data["location x"]) / 470).otherwise(data["location x"] / 470))
# for location y, we only need to divide by the max court length
data = data.withColumn("location y", data["location y"] / 500)

# Process time to be the duration of the match in seconds then normalize it
# to range from 0 to 1
# 2880 represents the number of minutes in a 48 minute match (4 quarters of 12 minutes each)
split_time = split(col('time'), ':')
#minutes = split_time.getItem(0)
#seconds = split_time.getItem(1)

#seconds = split_time.getItem(0).cast(IntegerType()) * 60 + split_time.getItem(1).cast(IntegerType())
#data = data.withColumn("time", seconds + ((data['quarter'] - 1) * 720) / 2880)
#data = data.withColumn("time", (((split(col('time'), ':').getItem(0).cast(IntegerType())) * 60)
#    + split(col('time'), ':').getItem(1).cast(IntegerType())) * ((data["quarter"] - 1) * 720) 
#    / 2880)
minutes = regexp_extract(col('time'), r'^(\d+):', 1).cast('int')
seconds = regexp_extract(col('time'), r':(\d+)$', 1).cast('int')
total_seconds = (minutes * 60) + seconds
data = data.withColumn('time', total_seconds)
data = data.drop("quarter")

# Show first 5 time rows
data.select("time").show(5)


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
