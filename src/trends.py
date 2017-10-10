import pytrends, datetime, pickle
from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import pyspark as ps
# from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
   'font.size'           : 20.0,
   'axes.titlesize'      : 'large',
   'axes.labelsize'      : 'medium',
   'xtick.labelsize'     : 'x-small',
   'ytick.labelsize'     : 'small',
   'legend.fontsize'     : 'xx-small',
    })


def get_trends(kw_list,timeframe='2017-03-01 2017-05-01'):
    pt = TrendReq()
    trends_dict = dict()

    for term in kw_list:
        pt.build_payload(kw_list=[term], timeframe=timeframe)
        trends_dict[term]=pt.interest_over_time()

    keywords=list(trends_dict.keys())

    carry = trends_dict[keywords[0]].merge(trends_dict[keywords[1]],left_index=True,right_index=True)

    for i in range(2,len(keywords)):
        carry = carry.merge(trends_dict[keywords[i]],left_index=True,right_index=True)

    return carry

def get_keywords(centroid):
    top_ten = []
    centroid_top_idx=np.argsort(-centroid)
    scale = centroid[centroid_top_idx]
    for i in range(10):
        top_ten.append(vocab[centroid_top_idx[i]])
    return scale, top_ten

def hour_day(x):
    return x//24

def assign_timeframe(date, cut=datetime.date(2017,3,1)):
    yyyy,MM,dd = date.split('-')
    ts = datetime.date(int(yyyy),int(MM),int(dd))
    return ((ts-cut).days)

if __name__ == '__main__':
    #load centroids
    with open('../data/centroids.pkl','rb') as f:
        centroids = pickle.load(f)

    # load vocab
    with open('../data/vocab.pkl','rb') as f:
        vocab = pickle.load(f)

    #one centroid
    for i in range(25):
        centroid= centroids[i]

        scale, top_ten = get_keywords(centroid)

        trends_df = get_trends(top_ten)

        trends_df.to_csv('topic_{}_trends.csv'.format(i))
