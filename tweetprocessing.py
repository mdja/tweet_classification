import re
import string

import pandas as pd
import numpy as np

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

class TweetProcessing():
    user_pattern = r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)'
    url_pattern = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})'    
    url_pattern2 = r'https://t.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,}'
    #url_pattern = '^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$'
    digit_pattern = r'^\d+\s|\s\d+\s|\s\d+$'
    rt_pattern = r'RT\s:*'
    additional_stopwords = ['cc ', 'cc:', 'cc.', 'a', 'd', 'g', 'e', 'y', 'ga', 'gmn', 'tdk', 'nah', 'sih', 'blm', 'ni', 'di', 'sy', 'sya', 'rt', 'jl', 'jl.', 'jln', 'jln.', 'no', 'no.', 'dlm', 'tx', 'thx', 'he', 'd', 'k', 'sm']
    
    
    def __init__(self, tweet):
        self.tweet = tweet
        stop_factory = StopWordRemoverFactory().get_stop_words()
        stop_factory = stop_factory + self.additional_stopwords
        dictionary = ArrayDictionary(stop_factory)
        self.strword = StopWordRemover(dictionary)
        
    def set_tweet(self, tweet):
        self.tweet = tweet
        
    def get_tweet(self, tweet):
        return self.tweet
            
    def clean_up_tweet_usernames(self):
        return re.sub(self.user_pattern, '', self.tweet)
    
    def clean_up_tweet_url(self):
        self.tweet = re.sub(self.url_pattern, '', self.tweet)
        self.tweet = self.tweet.replace("https://t.?", '')
        self.tweet = self.tweet.replace("https://t?", '')
        self.tweet = self.tweet.replace("https://?", '')
        return re.sub(self.url_pattern2, '', self.tweet)
    
    def clean_up_tweet_rt(self):
        return re.sub(self.rt_pattern,'', self.tweet)
    
    def clean_up_tweet_digits(self):
        self.tweet = ''.join([i for i in str(self.tweet) if not i.isdigit()])
        return self.tweet
        #return re.sub(self.digit_pattern,'', self.tweet)
    
    def remove_stop_words(self):                 
        self.tweet = self.strword.remove(self.tweet)
        return self.tweet
    
    def stemming_tweet(self):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        self.tweet = stemmer.stem(self.tweet)
        return self.tweet

    def clean_up_tweet(self):
        self.tweet = self.tweet.lower()
        #self.tweet = self.clean_up_tweet_usernames()
        self.tweet = self.clean_up_tweet_url()
        self.tweet = self.clean_up_tweet_rt()
        self.tweet = self.clean_up_tweet_digits()
        #self.tweet = self.tweet.replace('.',' ')
        self.tweet = self.tweet.replace(',',' ')
        self.tweet = self.tweet.replace('?','')
        self.tweet = self.tweet.replace('  ',' ')
        self.tweet = self.stemming_tweet()
        self.tweet = self.remove_stop_words()
        self.tweet = self.tweet.translate(string.punctuation)
        if self.tweet.startswith('"') and self.tweet.endswith('"'):
            self.tweet = self.tweet[1:-1]

        return self.tweet


class TweetDataLoader():
    def __init__(self, fdataname):
        self.fdataname = fdataname

    # load and split data from original csv files
    def load_split_data(self):   
        data = pd.read_csv(self.fdataname)
        data_numpy = data.fillna(0).as_matrix()
        
        tweet = data_numpy[:,2]
        keluhan = data_numpy[:,3]
        respon = data_numpy[:,4]
        
        Yvalue = []
        
        for i in range(len(data_numpy)):
            if keluhan[i] !=0:
                Yvalue.append(1)
            elif respon[i] != 0:
                Yvalue.append(2)
            else:
                Yvalue.append(3)
        
        Yvalue = np.asarray(Yvalue)
        return (tweet, Yvalue)
    

    def write_processed_csv(self, foutname, xvalues, yvalues):
        list_element = []
        index_to_delete = []
        for i in range(len(xvalues)):
            if  xvalues[i] in list_element:
                index_to_delete.append(i) 
    #         print("duplicate index ke " + str(i))
            else:
                list_element.append(xvalues[i])
        
        #remove duplicate tweet
        if (len(index_to_delete) > 0):
            xvalues = np.delete(xvalues,index_to_delete)
            yvalues = np.delete(yvalues,index_to_delete) 
    
        raw_data = { 'kelas' : yvalues,
            'tweet' : xvalues                    
        }
        df = pd.DataFrame(raw_data, columns = ['kelas','tweet'])
        df.to_csv(foutname,index=False)
 