#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 03:23:52 2019

@author: jamal
"""

import argparse
import string
import pandas as pd
import numpy as np
import pickle

import tweetprocessing

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers


def load_split_data(fname):   
    data = pd.read_csv(fname)
    data_numpy = data.fillna(0).as_matrix()
        
    kelas = data_numpy[:,0]
    tweet = data_numpy[:,1]     
        
    return (tweet, kelas)

def get_tweets():
    tweets = []
    if (args["tweet"] is not None):
        #print(args["tweet"])
        tweets.append(str(args["tweet"]))        
    elif (args["file"] is not None):
        with open(args["file"],'r') as file:
            for line in file:
                tweets.append(line)
    
    tweet_processing = tweetprocessing.TweetProcessing('')
    for i in range(len(tweets)):
        #print(tweet[i])
        tweet_processing.set_tweet(tweets[i])
        tweets[i] = tweet_processing.clean_up_tweet()
    
    return tweets

def make_prediction(tweet, filename): 
    foutname = 'test_out.txt'
    f = open(foutname, 'w')
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(tweet)

    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
    tfidf_vect.fit(tweet)

    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=10000)
    tfidf_vect_ngram.fit(tweet)

    loaded_model = joblib.load(filename)
    test_tweets = get_tweets()
    countv_new_tweet = count_vect.transform(test_tweets)
    #tfidf_new_tweet = tfidf_vect.transform(test_tweets)
    #tfidf_ngram_new_tweet = tfidf_vect_ngram.transform(test_tweets)
    # gunakan hanya count vector 
    kelas_tweet = loaded_model.predict(countv_new_tweet)
    for i in range(len(kelas_tweet)):
        kls = kelas_tweet[i]
        print("{0} -> {1}".format(test_tweets[i], category[kls]))
        f.write("{0} -> {1}\n".format(test_tweets[i], category[kls]))
    f.close()

def make_prediction_all(tweet):
    foutname = 'test_out.txt'
    f = open(foutname, 'w')
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(tweet)

    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
    tfidf_vect.fit(tweet)

    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=10000)
    tfidf_vect_ngram.fit(tweet)

    class_name = ['naive bayes', 'random forest', 'svm', 'xgboost']
    files = ['naive_bayes_model.dat','random_forest_model.dat','svm_model.dat','xgboost_model.dat']
    
    test_tweets = get_tweets()
    countv_new_tweet = count_vect.transform(test_tweets)
    kelas_tweet = []
    for i in range(len(class_name)):
        loaded_model = joblib.load(files[i])
        # gunakan hanya count vector 
        kelas_tweet.append(loaded_model.predict(countv_new_tweet))
    for i in range(len(test_tweets)):
        line = "{0} -> ".format(test_tweets[i])
        for j in range(len(class_name)):
            kls = kelas_tweet[j][i]
            line = line + " {0} : {1} ".format(class_name[j], category[kls])
            if (j < len(class_name) - 1):
                line += ","
        print(line)
        f.write(line + "\n")
    f.close()
        
parser = argparse.ArgumentParser("tweet_kelas")
parser.add_argument("-c", "--classifier", help="tipe classifier : (nbayes, rforest, svm, xgboost).", required=True)
parser.add_argument("-t", "--tweet", help="input file name.")
parser.add_argument("-f", "--file", help="input file name.")
args = vars(parser.parse_args())
# print("type = {}".format(args["classifier"]))
# print("tweet = {}".format(args["tweet"]))
# print("file = {}".format(args["file"]))

category = ['keluhan','respon','bukan keluhan/respon']
(tweet, kelas) = load_split_data('processed_tweets2.csv')

#for i in range(len(tweet)):
#    print(str(kelas[i]) + ', ' + tweet[i])

#loaded_model = joblib.load(filename)
#test_tweet = ['@adammaulud: didinyamah kieu ?? bosss! kumaha pak? @ridwankamil daerah sapharuhai',
#              '@ridwankamil tolong dong pak transportasi',
#              'pak, tadi habis kebun binatang, kebersihannya kurang orangutannya pak, mohon solusi pak.. @ridwankamil'
#              ]
#tfidf_new_tweet = tfidf_vect.transform(test_tweet)
#kelas_tweet = loaded_model.predict(tfidf_new_tweet)
#for kelas_item in kelas_tweet:
#    print("NB, category ", category[kelas_item])
#    

if (str(args["classifier"]).lower() == "nbayes"):
    filename = "naive_bayes_model.dat"
    make_prediction(tweet,filename)
elif (str(args["classifier"]).lower() == "rforest"):
    filename = "random_forest_model.dat"
    make_prediction(tweet,filename)
elif (str(args["classifier"]).lower() == "svm"):
    filename = "svm_model.dat"
    make_prediction(tweet,filename)
elif (str(args["classifier"]).lower() == "xgboost"):
    filename = "xgboost_model.dat"
    make_prediction(tweet,filename)
elif (str(args["classifier"]).lower() == "all"):
    make_prediction_all(tweet)
     

