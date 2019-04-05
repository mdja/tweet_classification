import tweetprocessing

fin_name = 'Ica_Labelled Tweets.csv'
fout_name = 'processed_tweets2.csv'

data_loader = tweetprocessing.TweetDataLoader(fin_name)
(tweet, kelas) = data_loader.load_split_data()
# text processing

tweet_processing = tweetprocessing.TweetProcessing('')
for i in range(len(tweet)):
    #print(tweet[i])
    tweet_processing.set_tweet(tweet[i])
    tweet[i] = tweet_processing.clean_up_tweet()

    
data_loader.write_processed_csv(fout_name, tweet, kelas)