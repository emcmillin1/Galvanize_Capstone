# Ask Me About My Website #

## Short AutoBiography

My name is Eric McMillin. Over the last couple years I graduated from CSU with a degree in physiology, ran a healthy lifestyles program for the Boys and Girls Club of Larimer county, ran the Children's Center portion of a Ski School, then applied for and got accepted into Galvanize' Data Science Immersive... This is my capstone project.

## Initial Direction

Over the course of the DSI immersive we learned about a lot of topics/ themes/ technologies. My interest in these ranged anywhere from "Cool, but not for me" to "I didn't think you were allowed to do that unless you worked for the NSA or had a PHD in four or five subjects"... and Natural Language Processing (NLP) and Web Scraping both fell into the later, so I wanted to look for a capstone under that umbrella.

## Project Description

As stated earlier I was looking for a project under the Web Scraping/ NLP umbrella, and was lucky enough to find a company that provides analytical tools to online writers and was willing to share their data with me. Luckily the company had a lot of data, and was very open to a wide array of ideas. After evaluating the desires of the company, the conditions of the data, and folding in my interests I decided to use unsupervised topic modeling in an to attempt to elucidate patterns tied to each topic. This would allow the company to make more educated decisions on advertising campaigns, and better recommendations by inferring information about new articles.

## Data

I was given click level data from one of my sponsor's clients. This data included information about the article, the user/ session, and the revenue collected at that level. From this I was able to strip out all of the unique urls (~300,000), to collect the articles that would be used in the topic modeling process. Later, I would fold in the rest of the metrics to perform aggregate analysis in order to answer my question "How can we use latent 'Topics' to provide more insight and make better decisions?""

# System Layout


<img alt="Capstone Layout" src="data/Capstone_Layout.png" width='1000' height = '600'>

## Departments

  I'm calling each of the shaded in areas on the diagram 'Departments'. Most departments worked on AWS clusters. Each of the pieces interacted with s3 in the same way. I wrote bash scripts to start a cluster, install dependencies, pull information to cluster, execute python scripts, the send output back to s3.

### Department 1: Scraping

  -Used [requests]('http://docs.python-requests.org/en/master/') and beautifulsoup to scrape 300,000 online articles. Max rate 30,000 articles per hour on 3: large clusters on AWS.

  #### Sample HTML

    <html>
      <body>

        <h1>My First Heading</h1>

        <p>My first paragraph.</p>

      </body>
    </html>

  #### BeautifulSoup output

      Beautiful Soup finds desired tag(<p>):
      <p>My first paragraph.</p>

      Output:
      My first Paragraph


### Department 1.1: Cleaning

  -Used regular expressions, stripped unicode, punctuation, sent cleaned text to s3 bucket

  -- Notes:

      Kept "'" because it shows up in a lot of stopwords and I was getting a lot of occurences of standalone
      t's and s's (ie. isn't > isn and t)

  #### Cleaning examples



### NLP sandbox
  -Used NLTK lemmatizer, pysparks default stopwords (adding in some of my own), attempted word2vec to drop out synonyms (failed)

  #### NLP examples



### Department 2: Clustering

  Pipeline(cleanish text from scraper process> tokens > dropped out stopwords > count vectorizer > TFIDF> kmeans (fit on TFIDF vectors))

    explain TF-IDF here

### Rational for Clustering

  Clusters (Topics) provide us a way to generalize article behaviors. (i.e. 2 articles in the same topic tend to have similar lifespans) we can use out fit pipeline to classify new articles into one of these topics, then make assumptions about how well it will do based on traits we see in that topic.



## Analysis

<img alt="topic0 revtime" src="img/topic0_revtime.png" width='1000' height = '600'>

<!-- <img alt="topic2 revtime" src="img/topic2_revtime.png" width='1000' height = '600'>

<img alt="topic18 revtime" src="img/topic18_revtime.png" width='1000' height = '600'> -->

<img alt="topic9dow" src="img/topic9_dow.png" width='1000' height = '600'>

<img alt="topic10dow" src="img/topic10_dow.png" width='1000' height = '600'>

<img alt="topic19dow" src="img/topic19_dow.png" width='1000' height = '600'>
