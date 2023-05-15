from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, mean, round, concat_ws, col, split, expr, regexp_extract
from pyspark.sql.types import IntegerType

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
data = data.drop("shoot player", 'date', 'home game', 'home team', 'away team', '#FtAtt', '#FtMade',
                 'self previous shot', 'opponent previous shot', 'time from last shot', 'Name', 'quarter', 'shot type')
# drop nulls in location x
data = data.dropna(subset=['location x'])
# drop players with 0 in Fg2PtAtt or Fg3PtAtt
data = data.filter((data["#Fg2PtAtt"] != 0) & (data["#Fg3PtAtt"] != 0))

# if points = 2 then accuracy = Fg2PtMade / Fg2PtAtt, else acuracy = Fg3PtMade / Fg3PtAtt
data = data.withColumn("accuracy", when(data["points"] == 2, data["#Fg2PtMade"] / data["#Fg2PtAtt"]).otherwise(data["#Fg3PtMade"] / data["#Fg3PtAtt"]))
# drop the columns that are not needed anymore [#Fg2PtAtt, #Fg2PtMade, #Fg3PtAtt, #Fg3PtMade]
data = data.drop("#Fg2PtAtt", "#Fg2PtMade",  "#Fg3PtAtt", "#Fg3PtMade")

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

data = data.fillna({'#Age': mean_age, '#Weight': mean_weight,
                     '#Height': mean_height})

# Normalize height using min normalization
data = data.withColumn("#Height", (data["#Height"] - data.selectExpr("min(`#Height`) as min").collect()[0][0]) / (data.selectExpr("max(`#Height`) as max").collect()[0][0] - data.selectExpr("min(`#Height`) as min").collect()[0][0])) 

# Normalize Age, Weight, GamesPlayed using Z-Score
data = data.withColumn("#Age", (data["#Age"] - mean_age) / data.selectExpr("stddev_samp(`#Age`) as std").collect()[0][0])
data = data.withColumn("#Weight", (data["#Weight"] - mean_weight) / data.selectExpr("stddev_samp(`#Weight`) as std").collect()[0][0])
data = data.withColumn("#GamesPlayed", (data["#GamesPlayed"] - mean_weight) / data.selectExpr("stddev_samp(`#GamesPlayed`) as std").collect()[0][0])

# Normalize location x and location y
# for location x, we'll split the full length of the court into one half-court
# and then normalize the values to be between 0 and 1
data = data.withColumn("location x", when(data["location x"] > 470, (940 - data["location x"]) / 470).otherwise(data["location x"] / 470))
# for location y, we only need to divide by the max court length
data = data.withColumn("location y", data["location y"] / 500)


# Encoding the categorical columns

# Current shot outcome encoding, SCORED: 1, otherwise: 0
data = data.withColumn("current shot outcome", when(data["current shot outcome"] == "SCORED", 1).otherwise(0))
# Rookie year encoding, Rookie: 1, otherwise: 0
data = data.withColumn("#Rookie", when(data["#Rookie"] == "Y", 1).otherwise(0))
# Map player position from ['SF' 'C' 'SG' 'PG' 'PF' 'G' 'F'] to [0, 1, 2, 3, 4 ,5, 6]
position_map = {'SF': '0', 'C': '1', 'SG': '2', 'PG': '3', 'PF': '4', 'G': '5', 'F': '6'}
data = data.replace(position_map, subset=['player position'])
data = data.withColumn("player position", data["player position"].cast(IntegerType()))


# We're only going to take into account the 10 most common shot types as there are
# around 800+ different shot types some of which only occur once or twice

# Get the 10 most common shot types
#shot_type = data.groupBy("shot type").count().orderBy("count", ascending=False).limit(10).collect()
# Map the shot types to integers
#shot_type_map = {shot_type[i][0]: str(i) for i in range(len(shot_type))}
# Replace the shot types with the integers
#data = data.replace(shot_type_map, subset=['shot type'])
#data = data.withColumn("shot type", data["shot type"].cast(IntegerType()))
# drop nulls after taking the 10 most common shot types
#data = data.dropna(subset=['shot type'])

# train gbt regressor model
# split data into training and testing sets

# create the feature vector
assembler = VectorAssembler(inputCols=["#Age", "#Height", "#Weight", 
    "#GamesPlayed", "location x", "location y", "player position",
    "time", "accuracy", '#Rookie' ], outputCol="features") 

# transform the data
data = assembler.transform(data)

train, test = data.randomSplit([0.75, 0.25], seed=42)

# create the gbt regressor model
gbt = GBTRegressor(featuresCol="features", labelCol="current shot outcome", maxIter=10)

# train the model
model = gbt.fit(train)

# make predictions on the test data
predictions = model.transform(test)

# evaluate the model
evaluator = RegressionEvaluator(labelCol="current shot outcome", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
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
