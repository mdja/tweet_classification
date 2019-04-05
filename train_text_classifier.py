import pandas as pd
import numpy as np
import pickle

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.externals import joblib

import xgboost



class TweetClassifier():
    '''    
    def __init__(self, tweet_train=[], tweet_test=[], kelas_train=[], kelas_test=[]):
        self.tweet_train = tweet_train
        self.tweet_test = tweet_test
        self.kelas_train = kelas_train
        self.kelas_test = kelas_test
    '''        
    def train_model(self,classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
    
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
    
        if is_neural_net:
            predictions = predictions.argmax(axis=-1)
    
        return metrics.accuracy_score(predictions, valid_y)
            
    # load data from processsed_tweets.csv

def load_split_data(fname):   
    data = pd.read_csv(fname)
    data_numpy = data.fillna(0).as_matrix()
        
    kelas = data_numpy[:,0]
    tweet = data_numpy[:,1]     
        
    return (tweet, kelas)

category = ['keluhan','respon','bukan keluhan/respon']

(tweet, kelas) = load_split_data('processed_tweets2.csv')
#for i in range(len(tweet)):
#    print(str(kelas[i]) + ', ' + tweet[i])

tweet_train, tweet_test, kelas_train, kelas_test = model_selection.train_test_split(tweet, kelas, test_size=0.33, random_state=42)
#print(Tweet_train)
encoder = preprocessing.LabelEncoder()
kelas_train = encoder.fit_transform(kelas_train)
kelas_test = encoder.fit_transform(kelas_test)


####Count vector
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(tweet)

# transform the training and validation data using count vectorizer object
tweet_train_count = count_vect.transform(tweet_train)
tweet_test_count = count_vect.transform(tweet_test)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
tfidf_vect.fit(tweet)
tweet_train_tfidf = tfidf_vect.transform(tweet_train)
tweet_test_tfidf = tfidf_vect.transform(tweet_test)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=10000)
tfidf_vect_ngram.fit(tweet)
tweet_train_tfidf_ngram =  tfidf_vect_ngram.transform(tweet_train)
tweet_test_tfidf_ngram =  tfidf_vect_ngram.transform(tweet_test)


tweet_classifier = TweetClassifier()

###################### naive Bayes #####################
nbayes_model = naive_bayes.MultinomialNB()
## Naive Bayes on Count Vectors
accuracyCount = tweet_classifier.train_model(nbayes_model, tweet_train_count, kelas_train, tweet_test_count, kelas_test)
print("NB, Count Vectors: ", accuracyCount)
filename = "naive_bayes_model.dat"
joblib.dump(nbayes_model, filename)

## Naive Bayes on Word Level TF IDF Vectors
accuracyTFIDF = tweet_classifier.train_model(nbayes_model, tweet_train_tfidf, kelas_train, tweet_test_tfidf, kelas_test)
print("NB, WordLevel TF-IDF: ", accuracyTFIDF)
#
accuracyNGram = tweet_classifier.train_model(nbayes_model, tweet_train_tfidf_ngram, kelas_train, tweet_test_tfidf_ngram, kelas_test)
print("NB, N-Gram Vectors: ", accuracyNGram)


##################### Random Forest ##################

modelRF = ensemble.RandomForestClassifier(n_jobs=2, random_state=0)
## RF on Count Vectors
accuracyCount = tweet_classifier.train_model(modelRF, tweet_train_count, kelas_train, tweet_test_count, kelas_test)
print("RF, Count Vectors: ", accuracyCount)
filename = "random_forest_model.dat"
joblib.dump(modelRF, filename)

## RF on Word Level TF IDF Vectors
accuracyTFIDF = tweet_classifier.train_model(modelRF, tweet_train_tfidf, kelas_train, tweet_test_tfidf, kelas_test)
print("RF, WordLevel TF-IDF: ", accuracyTFIDF)

## RF on N-Gram TF IDF Vectors
accuracyNGram = tweet_classifier.train_model(modelRF, tweet_train_tfidf_ngram, kelas_train, tweet_test_tfidf_ngram, kelas_test)
print("RF, N-Gram Vectors: ", accuracyNGram)

    
######################### SVM #####################
## SVM on Count Vectors
svm_model = svm.SVC(gamma='scale', decision_function_shape='ovo')
accuracyCount = tweet_classifier.train_model(svm_model, tweet_train_count, kelas_train, tweet_test_count, kelas_test)
print("SVM, Count Vectors:: ", accuracyCount)
filename = "svm_model.dat"
joblib.dump(svm_model, filename)
#
## SVM on Word Level TF IDF Vectors
accuracyTFIDF = tweet_classifier.train_model(svm_model, tweet_train_tfidf, kelas_train, tweet_test_tfidf, kelas_test)
print("SVM, WordLevel TF-IDF: ", accuracyTFIDF)
#
## SVM on Ngram Level TF IDF Vectors
accuracyNGram = tweet_classifier.train_model(svm_model, tweet_train_tfidf_ngram, kelas_train, tweet_test_tfidf_ngram, kelas_test)
print("SVM, N-Gram Vectors: ", accuracyNGram)

######################## Gradient Boosting################
## Extereme Gradient Boosting on Word Level TF IDF Vectors
xgb_model = xgboost.XGBClassifier(max_depth=8,
                           min_child_weight=1,
                           learning_rate=0.1,
                          n_estimators=500,
                          silent=True,
                          objective="multi:softprob",
                          gamma=0,
                          max_delta_step=0,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          reg_alpha=0,
                          reg_lambda=0,
                          scale_pos_weight=1,
                          seed=1,
                          missing=None)
accuracyCount = tweet_classifier.train_model(xgb_model, tweet_train_count.tocsc(), kelas_train, tweet_test_count.tocsc(), kelas_test)
print("Xgb, WordLevel count : ", accuracyCount)
filename = "xgboost_model.dat"
joblib.dump(xgb_model, filename)

accuracyTFIDF = tweet_classifier.train_model(xgb_model, tweet_train_tfidf.tocsc(), kelas_train, tweet_test_tfidf.tocsc(), kelas_test)
print("Xgb, WordLevel TF-IDF: ", accuracyTFIDF)
#
accuracyNGram = tweet_classifier.train_model(xgb_model, tweet_train_tfidf_ngram.tocsc(), kelas_train, tweet_test_tfidf_ngram.tocsc(), kelas_test)
print("Xgb, N-Gram Vectors: ", accuracyNGram)


 

