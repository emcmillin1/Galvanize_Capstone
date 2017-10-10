import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
from string import punctuation, printable
from time import sleep



class Scraper():
    def __init__(self, in_path, out_path, subset=False):
        self.in_path = in_path
        self.out_path = out_path
        self.punctuation = set(punctuation)
        self.punctuation.remove("'")
        self.punctuation.remove(".")
        self.df = pd.read_csv(self.in_path)
        if subset:
            self.subset=subset
            self.df=self.df[subset[0]:subset[1]]
        self.n_docs = self.df.shape[0]
        self.doc_counter = 0



    def to_csv(self, remove_empty=True):
        self.scrape()
        self.clean()
        if remove_empty:
            print('Removing Nans...')
            self.df=self.df[self.df.text!=np.nan]
        self.df.to_csv(self.out_path)

    def _scrape_one(self, url):
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
        #to spice up the command line tracking
        #get html
        if isinstance(url,str) != True:
            print('failed')
            return 'failed'

        response = requests.get(url)
        #check that request cleared
        if str(response.status_code)[0]!='2':
            print('failed')
            return 'failed'

        if response.is_redirect:
            print('redirect')
            return 'redirect'

        else:
            #convert response to soup
            soup = BeautifulSoup(response.text, 'html.parser')
            print("Scraping {}...".format(soup.title.string))
            #find all occurences of type 'tag'.  Change this based on needs
            content = soup.find_all('p')
            text = ''.join(p.get_text() for p in content)
            #Tracking completion status
            print('{}/{}.\tSubset: {}'.format(self.doc_counter,self.n_docs))
            self.doc_counter+=1
            return text


    def scrape(self):
        '''
        Applies Scrape_one over dataframe
        '''
        #sample syntax for when more than one column created by .apply
        # df.merge(df.one.apply(lambda row: pd.Series({'four':row+1,'five':row-1})),left_index=True, right_index=True)

        start = datetime.datetime.now()
        self.df['text'] = self.df.post_url.apply(lambda row: self._scrape_one(row))
        #record and drop request failures
        self.failures=self.df[self.df['text']=='failed']
        self.redirect=self.df[self.df['text']=='redirect']
        self.df=self.df[self.df['text']!='failed']
        self.df=self.df[self.df['text']!='redirect']
        exec_time = (datetime.datetime.now()-start)
        print('Successfully Scraped {} URLs in: {}'.format(self.df.shape[0],exec_time.seconds))

    def _clean_one(self, text):
        # First remove punctuation form string
        # .translate works differently from 2 to 3 so check version number
        '''
        Input: html content and set of stop_words to exclude (sklearn ENGLISH_STOP_WORDS)
        Output: stripped, lowered, lemmatized string
        Possible improvements: regex to recognize and replace abbreviated date with html timestring
        '''
        #spacy expects unicode
        # text = uni.encode(text.translate(' ', punctuation))

        # remove punctuation and unicode objects
        if isinstance(text, str) != True:
            return np.nan
        else:
            text = text.replace('\n', '').replace('&nbsp',' ').replace('\t', ' ')
            text = "".join([char.lower() if char in printable #unicode
                and char not in self.punctuation #punctuation
                # and char not in '0123456789' #numbers
                else ' ' for char in text]) #maintain Seperation
            text = ' '.join([chunk for chunk in text.split()]) # Replaces multispace with one space
            return text

    def clean(self):
        '''
        Applies clean_one over dataframe
        '''
        print('Cleaning Text...')
        start = datetime.datetime.now()

        self.df['text'] = self.df.text.apply(lambda row: self._clean_one(row))
        exec_time = (datetime.datetime.now()-start).seconds
        print("Completed cleaning {} documents in {} seconds".format(self.df.shape[0], exec_time))




if __name__=='__main__':
    s = Scraper('../data/', ''.join(['../data/articles_out/acticles_',str(i),'.csv']))
    s.to_csv()

    # for i in range(100,200):
    #     print(i)
    #     s = Scraper(''.join(['../data/100kurls/urls_',str(i),'.csv']), ''.join(['../data/articles_out/acticles_',str(i),'.csv']))
    #     s.to_csv()

    #one url to test that it works
    # s = Scraper('../data/one_url.csv', '../data/df_out.csv')
