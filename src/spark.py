# To be used after scraper.py
#depends on df_out
from collections import defaultdict
import pyspark as ps
from pyspark.ml.clustering import KMeans, LDA
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover, Tokenizer,Word2Vec
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import udf, col, split
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import Row
from string import punctuation, printable
import numpy as np
from nltk import WordNetLemmatizer

def lemma(tokens):
    return [wnl.lemmatize(token.replace('.','')) for token in tokens if not token.isnumeric()]



if __name__ == "__main__":
    print("Starting Spark Session...")
    spark = ps.sql.SparkSession.builder \
    .master("local[*]") \
    .appName("Article df to Topics") \
    .getOrCreate()

    #read csv into spark dataframe
    print('Reading in DataFrame...')
    df = spark.read.format("csv").option("header", "true").load('../data/id_text.csv')
    df = df.select('post_id','text')

    #split off keywords if desired
    df.createTempView('df')

    # trump_df = spark.sql('SELECT * FROM df WHERE df.text CONTAINS "trump"')
    # trump_df = trump_df.withColumn('topic',-1)
    # out = trump_df.select('post_id','topic')
    # out = out.toPandas()
    # out.to_csv('kard_topic.csv')
    #
    # df = spark.sql('SELECT * FROM df WHERE df.text NOT CONTAINS "Kardashian" AND NOT CONTAINS "Jenner")
    #tokenize string
    print('Tokenizing Text...')
    tokenizer = Tokenizer(inputCol='text', outputCol='tokens')
    df = tokenizer.transform(df)


    wnl = WordNetLemmatizer()
    print('Lemmatizing Text...')
    lemma_udf = udf(lambda row: lemma(row), ArrayType(StringType()))
    df = df.withColumn('lemmed_tokens', lemma_udf(df.tokens))


    # remove stopwords
    print('Removing Stop Words...')
    swr = StopWordsRemover(inputCol='lemmed_tokens', outputCol='filtered_tokens')
    stops = swr.loadDefaultStopWords('english')
    for stop in stops:
        stop.replace('’','')
    for word in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'q', 'r', 's' ,'t', 'u','v', 'w', 'x', 'y', 'z',
        'ha','wa','getty','image','ap','pictwittercom']:
        stops.append(word)
    swr.setStopWords(stops)
    df = swr.transform(df)

    df=df.select('post_id','filtered_tokens')



    print("Post Stop Word Remove")
    df.take(1)
    df.cache()

    #options for NLP here: word 2 vec to drop synonyms, n-grams, etc

    #tokens to counts
    print('Processing Through Count Vectorizer...')
    cv = CountVectorizer(inputCol='filtered_tokens', outputCol='counts', minDF=5,minTF=5,vocabSize=2000) #options for hyperparameters
    cvModel = cv.fit(df)
    df = cvModel.transform(df)
    vocab=cvModel.vocabulary

    #count vectors to tfidf
    print('Processing Through IDF...')
    idf = IDF(inputCol='counts', outputCol='tfidf') #options for hyperparameters
    idfModel = idf.fit(df)
    df = idfModel.transform(df)




    k_cost= dict()
    #cluster topics
    print('Assigning clusters...')
    for i in [25]:
        kmeans = KMeans(k=i, maxIter=50, seed=123,featuresCol='tfidf',predictionCol=''.join(['topic_k',str(i)]))
        kmeans_model = kmeans.fit(df)
        df = kmeans_model.transform(df)
        cost=    kmeans_model.computeCost(df)
        print(i, ': ',cost)
        k_cost[i]=cost/1000000

        # Track centroids
        group_sums = df.groupby(''.join(['topic_k',str(i)]))
        print(group_sums.count().show())
        centroids = kmeans_model.clusterCenters()

        topic_top_ten = defaultdict(list)
        topic_indexer=0

        # Top ten tokens per centroid
        print('Finding Centroid top Tokens...')
        for centroid in centroids:
           centroid_top_idx=np.argsort(-centroid)
           for i in range(10):
               topic_top_ten[topic_indexer].append(vocab[centroid_top_idx[i]]) #vocab (list) doesnt support multiindexing.. convert to array?
           print('Topic {} Top Tokens: \n\t{}\n'.format(topic_indexer,topic_top_ten[topic_indexer]))
           topic_indexer+=1

    print(k_cost)
    out = df.select('post_id','topic_k25').toPandas()
    out.to_csv('df_w_topics.csv')

    #LDA options
    # lda = LDA(k=5, featuresCol='tfidf', maxIter=20)
    # ldamodel=lda.fit(idf_df)
    #
    # #topics extrapolated from lda model (spark df)
    # topics=ldamodel.describeTopics(10)
