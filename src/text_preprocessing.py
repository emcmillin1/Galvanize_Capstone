#text preprocessing for capstone
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy, re, os, requests, sys
from string import punctuation, printable

if not 'nlp' in locals():
    print("Loading English Module...")
    nlp = spacy.load('en')




def scrape(url, tag, text=True):
    '''
    input:
        url: string
        tag: the html tag that BeautifulSoup will look for
        text: boolean, if text function uses .get_text() function
    output:
        if text = True; returns text contained within 'tag'
        else:  returns raw html
    improvements:
        more flexibility with tag queries
    '''

    #get html
    response = requests.get(url)
    # print(response.text)
    #convert response to soup
    soup = BeautifulSoup(response.text, 'html.parser')
    #find all occurences of type 'tag'.  Change this based on needs
    content = soup.find_all(tag)
    # print(content)
    if text:
        return ' '.join(p.get_text() for p in content)
    else:
        return content

def clean(content, stop_words):
    # First remove punctuation form string
    # .translate works differently from 2 to 3 so check version number
    '''
    Input: html content and set of stop_words to exclude (sklearn ENGLISH_STOP_WORDS)
    Output: stripped, lowered, lemmatized string
    Possible improvements: regex to recognize and replace abbreviated date with html timestring
    '''
    #remove non-breaking spaces
    content = content.replace('&nbsp;', ' ')

    #check system version, remove punctuation
    if sys.version_info.major == 3:

        PUNCT_DICT = {ord(punc): None for punc in punctuation}
        content = content.translate(PUNCT_DICT)
    else:
        # spaCy expects a unicode object
        content = unicode(content.translate(' ', punctuation))

    #get rid of line breaks
    content = content.replace('\n', '')

    # remove unicode
    clean_content = "".join([char for char in content if char in printable])

    # Run the content through spaCy
    content = nlp(clean_content)

    # Lemmatize and lower text
    tokens = [word_net.lemmatize(str(token).lower()) for token in content]

    #join and return output
    return ' '.join(w for w in tokens if w not in stop_words)


'''
Complications:
    nbsp;   \n  timestring objects  href's  twitter user reference
Improvements:
    Implement spacy,
Direction:
    vectorize for database application 
'''



if __name__ == '__main__':
    #url for page and tag for what html tag of interest
    # url, tag = 'http://www.lifedaily.com/teen-reunites-lost-toddler-with-family-during-flood/', 'p'
    url, tag = 'http://www.lifedaily.com/stolen-dog-immediately-recognizes-owner-in-court-room/', 'p'

    content = scrape(url, tag)
    # print(content)

    word_net = WordNetLemmatizer()
    clean_content = clean(content, ENGLISH_STOP_WORDS)
    print(clean_content)
