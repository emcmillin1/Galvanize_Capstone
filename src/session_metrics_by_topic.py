import pyspark as ps
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, FloatType
from pyspark.sql import Row
import numpy as np
import datetime
import pandas as pd



# if __name__ == '__main__':

print("Starting Spark Session...")
spark = ps.sql.SparkSession.builder \
.master("local[*]") \
.appName("Article df to Topics") \
.getOrCreate()

schema = StructType([
    StructField('uuid', StringType(), True),
    StructField('sessionId', StringType(), True),
    StructField('ts', StringType(), True),
    StructField('post_id', FloatType(), True),
    StructField('post_published', StringType(), True),
    StructField('post_url', StringType(), True),
    StructField('deviceType', StringType(), True),
    StructField('hit_revenue', FloatType(), True)])


#read csv into spark dataframe
print('Reading Metrics in DataFrame...')
metrics_df = spark.read.format("csv").option("header", "true").load('../data/data.csv', schema=schema)

#register table
metrics_df.createOrReplaceTempView('metrics')


def assign_timeframe(ts, cut=datetime.datetime(2017,3,1,0,0,0)):
    parts = ts.split()
    yyyy,MM,dd=    parts[0].split('-')
    hh,mm,ss = parts[1].split(':')
    ts = datetime.datetime(int(yyyy),int(MM),int(dd),int(hh),int(mm),int(ss))
    return ((ts-cut).total_seconds)

atf_udf = udf(lambda x: assign_timeframe(x),FloatType())

metrics_df = metrics_df.withColumn('timeframe',atf_udf(combined['ts']))

print('Reading in topics DataFrame')
topics_df =  spark.read.format("csv").option("header", "true").load('../data/id_topic.csv')

#register table
topics_df.createOrReplaceTempView('topics')

combined = spark.sql('''
    SELECT topic_k25 as topic, sessionId, metrics.post_id, timeframe, hit_revenue,deviceType
        FROM metrics JOIN topics ON topics.post_id=metrics.post_id''' )



combined.createOrReplaceTempView('combined')

session_starts=spark.sql('''
    SELECT DISTINCT(sessionId), post_id, topic
        FROM combined
        WHERE timeframe ==
            (SELECT min(timeframe)
                FROM combined GROUP BY sessionId)''' )

session_starts.createOrReplaceTempView('starts')

top_session_mets=spark.sql('''
    SELECT DISTINCT(starts.sessionId) as sessionId, starts.topic, sum(hit_revenue) as revenue, count(post_id) as views
        FROM combined
            JOIN starts ON DISTINCT(combined.sessionId)=starts.sessionId
        GROUP BY starts.topic, starts.sessionId''' )

top_session_mets.createOrReplaceTempView('sesh_top')

check = spark.sql('''SELECT sessionId, topic, revenue, views
                        FROM sesh_top
                            LIMIT 5''' )

print(check.show())

# out=top_session_mets.toPandas()
# out.to_csv('../data/session_mets_by_topic.csv')
#
#
# #avg bid by topic
# avg_bids= spark.sql('''
#     SELECT topic, avg(hit_revenue) as avg_hit
#         FROM combined
#         GROUP BY topic''' )
#
# out=avg_bids.toPandas()
# out.to_csv('../data/avg_hit_by_topic.csv')
#
#
# #demographic breakdown by topic
# smartphone_by_top= spark.sql('''
#     SELECT topic, count(deviceType) as avg_hit
#         FROM combined
#             WHERE deviceType='smartphone'
#         GROUP BY topic''' )
#
# desktop_by_top= spark.sql('''
#     SELECT topic, count(deviceType) as avg_hit
#         FROM combined
#             WHERE deviceType='desktop'
#         GROUP BY topic''' )
#
# topic_device = smartphone_by_top.join(desktop_by_top, desktop_by_top.topic == smartphone_by_top.topic, 'outer')
#
# out=avg_bids.toPandas()
# out.to_csv('../data/avg_hit_by_topic.csv')



# df = out.toPandas()
# df.to_csv('../data/binned_topic_df.csv')

# grouped.agg({(metrics_df.ts)})
