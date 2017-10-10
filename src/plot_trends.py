import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression

# In[2]:

import pyspark as ps
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row
from string import punctuation, printable
import numpy as np
from nltk import WordNetLemmatizer
from datetime import date
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import matplotlib as mpl



# In[3]:

spark = ps.sql.SparkSession.builder .master("local[3]") .appName("Article df to Topics") .getOrCreate()


# In[4]:

schema = StructType([
    StructField('c0',IntegerType(),True),
    StructField('topic', IntegerType(), True),
    StructField('timeframe', FloatType(), True),
    StructField('articles', IntegerType(), True),
    StructField('views', IntegerType(), True),
    StructField('revenue', FloatType(), True)])
#,topic,timeframe,articles,views,revenue


# In[5]:

metrics_df = spark.read.format("csv").option("header", "true").load('../data/binned_final.csv',schema=schema)


# In[6]:

mpl.rcParams.update({
   'font.size'           : 20.0,
   'axes.titlesize'      : 'large',
   'axes.labelsize'      : 'medium',
   'xtick.labelsize'     : 'x-small',
   'ytick.labelsize'     : 'small',
   'legend.fontsize'     : 'xx-small',
    })


# In[7]:


metrics_df = metrics_df.filter(metrics_df.timeframe>=0)


# In[10]:

metrics_df.createOrReplaceTempView('metrics')


# In[11]:

times = spark.sql('SELECT DISTINCT(timeframe) FROM metrics ORDER BY timeframe').toPandas()


# In[12]:

id_top=pd.read_csv('../data/id_topic.csv',index_col=0)


# In[13]:

id_pub=pd.read_csv('../data/id_pub.csv')


# In[14]:

topics_dict=dict()


# In[15]:

for i in range(25):
    topics_dict[i]=pd.DataFrame(spark.sql('SELECT timeframe,views,articles,revenue FROM metrics WHERE topic={}'.format(i)).collect(),columns=['timeframe','views','articles','revenue'])


# In[16]:

grouped=metrics_df.groupby('timeframe').agg({'views':'sum','revenue':'sum','articles':'sum'})

base = grouped.toPandas()

base.sort_values('timeframe',inplace=True)

base = base[['timeframe','sum(articles)','sum(views)','sum(revenue)']]

base.rename(columns={'sum(articles)':'articles','sum(views)':'views','sum(revenue)':'revenue'},inplace=True)


with open('../data/centroids.pkl','rb') as f:
    centroids = pickle.load(f)


# In[27]:

with open('../data/vocab.pkl','rb') as f:
    vocab = pickle.load(f)




avg_click=dict()
for i in range(25):
    z=topics_dict[i]
    avg_click[i]=(z.revenue.values.sum()/z.views.values.sum())
    z.sort_values('timeframe', inplace=True)



def hour_day(x):
    return x//24
to_hour_udf=udf(lambda x: hour_day(x),FloatType())

metrics_df=metrics_df.withColumn('day',to_hour_udf(metrics_df.timeframe))

top_day = metrics_df.groupby('day','topic').sum().toPandas()


#####################################
#may need
    # centroid = centroids[22]
    # centroid_top_idx=np.argsort(-centroid)
    # scale = centroid[centroid_top_idx]
    # for i in range(10):
    #     topic_top_ten[topic_indexer].append(vocab[centroid_top_idx[i]]) #vocab (list) doesnt support multiindexing.. convert to array?


# In[40]:
trend_dfs=dict()
for i in range(25):
    trend_dfs[i] = pd.read_csv('topic_{}_trends.csv'.format(i))


def assign_timeframe(date, cut=datetime.date(2017,3,1)):
    yyyy,MM,dd = date.split('-')
    ts = datetime.date(int(yyyy),int(MM),int(dd))
    return ((ts-cut).days)

for i in range(25):
    trend_dfs[i]['day'] = trend_dfs[i].date.apply(lambda x: assign_timeframe(x))


def get_keywords(centroid):
    top_ten = []
    centroid_top_idx=np.argsort(-centroid)
    scale = centroid[centroid_top_idx]
    for i in range(10):
        top_ten.append(vocab[centroid_top_idx[i]])
    return scale, top_ten

y_true = top_day[top_day.topic==17].sort_values('day')['sum(revenue)']
x_train = trend_dfs[17][['day','scott', 'kanye', 'blac', 'khloe', 'rob', 'chyna', 'kourtney', 'kim', 'kardashian']]
x_train = x_train[x_train['day']<=60]
model = LinearRegression()
model.fit(x_train,y_true)
y_pred = model.predict(x_train)
score = model.score(x_train,y_true)

fig, ax3 = plt.subplots(1,1,figsize=(10,10))
ax3.plot(range(61),y_true, lw=4, label='Actual Revenue')
ax3.plot(range(61),y_pred, lw=4, label='Predicted Revenue (Linear Model)')
ax3.set_title('Topic 17 Revenue Over Time vs.\nLinear Predictive Model')
ax3.set_xticks([0,7,14,21,28,35,42,49,56,63])
ax3.set_xticklabels(labels=['March-01','March-08','March-15','March-22','March-29','April-05','April-12','April-19','April-26'])
ax3.set_ylabel('Revenue ($)')
for tick in ax3.get_xticklabels():
    tick.set_rotation(-60)
ax3.legend()
plt.savefig('google_regression_line.png')
        # timeframes = base['timeframe']
#
# for i in range(25):
#     centroid=centroids[i]
#     scale, top_ten = get_keywords(centroid)
#     # print(top_ten)
#     # print(scale)
#     # for j in range(10):
#     #     print(scale[j],top_ten[j],trend_dfs[i][str(top_ten[j])]*scale[j])
#     print(i)
#     if i in [1,7]:
#         pass
#     else:
#         trend_dfs[i]['trendline'] = trend_dfs[i][top_ten[0]]*scale[0] +  trend_dfs[i][top_ten[1]]*scale[1] + trend_dfs[i][top_ten[2]]*scale[2] + trend_dfs[i][top_ten[3]]*scale[3] + \
#             trend_dfs[i][top_ten[4]]*scale[4] + trend_dfs[i][top_ten[5]]*scale[5] + trend_dfs[i][top_ten[6]]*scale[6] + trend_dfs[i][top_ten[7]]*scale[7] + \
#             trend_dfs[i][top_ten[8]]*scale[8] + trend_dfs[i][top_ten[9]]*scale[9]
#
# fig, ax3 = plt.subplots(1,1,figsize=(10,10))
# for i in [2,4,5]:
#     if i in [1,7]:
#         pass
#
#     else:
#         views = top_day[top_day['topic']==i][['day','sum(views)','sum(articles)','sum(revenue)']]
#         views.sort_values('day',inplace=True)
#
#         ax3.plot(views['day'],views['sum(revenue)']/views['sum(views)'],label='Topic {}'.format(i),lw=4,alpha=1)
#
#         # ax4 = ax3.twinx()
#         # ax4.plot(trend_dfs[i].day,trend_dfs[i].trendline/10)
#
# ax3.set_title('Average Revenue per Click Over Time')
# # ax3.set_ylim(.002,.0085)
# ax3.set_xticks([0,7,14,21,28,35,42,49,56,63])
# ax3.set_xticklabels(labels=['March-01','March-08','March-15','March-22','March-29','April-05','April-12','April-19','April-26'])
# ax3.set_ylabel('Revenue ($) per Page View')
# for tick in ax3.get_xticklabels():
#     tick.set_rotation(-60)
# ax3.legend()
#
#         # ax4.set_ylabel('Normalized Searches')
#
# plt.savefig('all_topics_revenue_over_time.png')
#
# for i in range(17,18):
#     if i in [1,7]:
#         pass
#
#     else:
#         fig, ax3 = plt.subplots(1,1,figsize=(10,10))
#         views = top_day[top_day['topic']==i][['day','sum(views)','sum(articles)','sum(revenue)']]
#         views.sort_values('day',inplace=True)
#
#         ax3.plot(views['day'],views['sum(revenue)'],lw=4,label='Topic {}'.format(i),color='darkorange',alpha=.6)
#
#         ax4 = ax3.twinx()
#         ax4.plot(trend_dfs[i].day,trend_dfs[i].trendline/10,lw=4)
#
#         ax3.set_title('Topic Revenue vs Term Trends')
#         ax3.set_xticks([0,7,14,21,28,35,42,49,56,63])
#         ax3.set_xticklabels(labels=['March-01','March-08','March-15','March-22','March-29','April-05','April-12','April-19','April-26'])
#         ax3.set_ylabel('Revenue ($)')
#         for tick in ax3.get_xticklabels():
#             tick.set_rotation(-60)
#         ax3.legend()
#
#         ax4.set_ylabel('Normalized Searches')
#
#         plt.savefig('topic_{}_vs_trends_revenue.png'.format(i))
#
# for i in range(17,18):
#     if i in [1,7]:
#         pass
#
#     else:
#         fig, ax3 = plt.subplots(1,1,figsize=(10,10))
#         views = top_day[top_day['topic']==i][['day','sum(views)','sum(articles)','sum(revenue)']]
#         views.sort_values('day',inplace=True)
#
#         ax3.plot(views['day'],views['sum(views)'],lw=4,label='Topic {}'.format(i),color='darkorange',alpha=.6)
#
#         ax4 = ax3.twinx()
#         ax4.plot(trend_dfs[i].day,trend_dfs[i].trendline/10, lw = 4)
#
#         ax3.set_title('Topic Views vs Term Trends')
#         ax3.set_xticks([0,7,14,21,28,35,42,49,56,63])
#         ax3.set_xticklabels(labels=['March-01','March-08','March-15','March-22','March-29','April-05','April-12','April-19','April-26'])
#         ax3.set_ylabel('Views')
#         for tick in ax3.get_xticklabels():
#             tick.set_rotation(-60)
#         ax3.legend()
#
#         ax4.set_ylabel('Normalized Searches')
#
#         plt.savefig('topic_{}_vs_trends_views.png'.format(i))
# # In[ ]:
#
# n_times=base.shape[0]
# z=topics_dict[1]
#
# revenue, views, articles= np.zeros(n_times),np.zeros(n_times),np.zeros(n_times)
# for i in list(times['timeframe'].values):
#     try:
#         row= z[z['timeframe']==i]
#         print(row)
#         idx=int(i/4)
#         revenue[idx]=row['revenue'].values[0]
#         views[idx]=row['views'].values[0]
#         articles[idx]=row['articles'].values[0]
#     except:
#         revenue[idx]=0
#         views[idx]=0
#         articles[idx]=0
#     z_out=pd.DataFrame([timeframes]).T #,revenue,views,articles
#     z_out['revenue']=revenue
#     z_out['views']=views
#     z_out['articles']=articles
#
#
# # In[ ]:
#
# top22 = topics_dict[22]
#
#
# # In[ ]:
#
# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# ss.fit(top22[['revenue','views','articles']])
# top22[['revenue','views','articles']] = ss.transform(top22[['revenue','views','articles']])
#
#
# # In[ ]:
#
# from datetime import date
# def to_date(x):
#     d = date
#     return d.fromordinal(736388+int(x))
#
#
# # In[ ]:
#
# top22['date']=top22.timeframe.apply(lambda x: to_date(x))
#
#
# # In[ ]:
#
# top22[['date','views']].to_csv('topic_23.csv')
#
#
# # In[ ]:
#
# who
#
#
# # In[ ]:
#
# rev_view = metrics_df.groupBy('topic').agg({'revenue':'sum','views':'sum'})
#
#
# # In[ ]:
#
# rev_view.take(1)
#
#
# # In[ ]:
#
# rev_view = rev_view.toPandas()
#
#
# # In[ ]:
#
# rev_view.sort_values('topic',inplace=True)
#
#
# # In[ ]:
#
# rev_view.head()
#
#
# # In[ ]:
#
# fig, ax1 = plt.subplots(1,1,figsize=(10,10))
# ax1.bar(rev_view['topic'],rev_view['sum(revenue)']/rev_view['sum(views)'])
# ax1.set_xticks(range(25));
# for tick in ax1.get_xticklabels():
#     tick.set_rotation(-90)
# ax1.set_xlabel('Topic Index')
# ax1.set_ylabel('Revenue Per View')
# ax1.set_title('Revenue Per Click by Topic')
# plt.savefig('revenue_per_click.png')
#
#
# # In[ ]:
#
# rev_view['prop']= np.random.random(25)
# rev_view['prop1']= rev_view['prop'].apply(lambda x: 1-x)
#
#
# # In[ ]:
#
# fig, ax1 = plt.subplots(1,1,figsize=(10,10))
# ax1.bar(rev_view['topic'],rev_view['prop'],label='Iphone')
# ax1.bar(rev_view['topic'],rev_view['prop1'],bottom=rev_view['prop'],label='others')
# ax1.set_xticks(range(25));
# for tick in ax1.get_xticklabels():
#     tick.set_rotation(-90)
# ax1.set_xlabel('Topic Index')
# ax1.set_ylabel('Proportion')
# ax1.legend()
# ax1.set_title('Proportion IPhone Users Per Topic')
# plt.savefig('prop_iphone_per_topic.png')
#
#
# # In[ ]:
#
#
#
#
# # In[ ]:
#
#
#
#
# # In[ ]:
#
#
#
