import re
import nltk
import json
import gzip
import os
import pprint
from datetime import datetime
from tqdm import tqdm
from dateutil.parser import parse
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, CursorNotFound
import pymongo

class TweetPreprocessor():
    def __init__(self, tweet_data_path):
        self.tweet_data_path = tweet_data_path
        self.files = os.listdir(self.tweet_data_path)
        self.client = MongoClient()
        self.db_metoo_tweets = self.client["new_metoo"]
        self.metoo_tweets = self.db_metoo_tweets.metoo_tweets

    def get_new_client(self):
        self.client = MongoClient()
        self.db_metoo_tweets = self.client["new_metoo"]
        self.metoo_tweets = self.db_metoo_tweets.metoo_tweets

    def push_tweets_to_db(self):
        tweets = self.read_tweets(self.files)
        i=0
        user_data = {}
        for tweet in tqdm(tweets):
            cur_tweet_data, user_data = self.process_tweet(tweet, user_data)
            i += 1
            #     if i%10000==0:
            #         print(i)
            if cur_tweet_data is not None:

                try:
                    temp = self.metoo_tweets.insert_one(cur_tweet_data)
                #             temp = metoo_tweets.find_one_and_update({"tweet_id" : str(cur_tweet_data["tweet_id"])}, {"$set":
                #         {"favoritesCount": cur_tweet_data["favoritesCount"]}
                #     },upsert=True)
                #             print(temp["favoritesCount"])
                #             if temp["favoritesCount"]==0:
                #                 print(tweet["link"])
                #                 print(temp["tweet_id"])

                except:
                   self.get_new_client()
                    # temp2 = metoo_tweets.find_one_and_update({"tweet_id" : str(cur_tweet_data["tweet_id"])}, {"$set":
            # {"favoritesCount": cur_tweet_data["favoritesCount"]}})

    def read_tweets(self, files):
        n_files = len(files)
        for i, file in enumerate(files):
            # print(f'Working on file: {i} of {n_files}')
            with gzip.open(os.path.join(self.tweet_data_path, file), 'rb') as f:
                for line in f:
                    line = json.loads(line)
                    yield (line)

    def process_user_data(self, tweet_dat, user_data):

        cur_user_id = tweet_dat['actor']['id'].split(':')[-1]

        try:  # See if user data already stored

            user_data[cur_user_id]['n_corpus_tweets'] += 1
            return (user_data)

        except KeyError as e:  # If not stored, get fields

            cur_user_data = {}
            cur_user_data['user_id'] = cur_user_id
            cur_user_data['verb'] = tweet_dat['verb']
            cur_user_data['user_objectType'] = tweet_dat['actor']['objectType']
            cur_user_data['displayName'] = tweet_dat['actor']['displayName']
            cur_user_data['summary'] = tweet_dat['actor']['summary']
            cur_user_data['verified'] = tweet_dat['actor']['verified']
            cur_user_data['statusesCount'] = tweet_dat['actor']['statusesCount']
            cur_user_data['favoritesCount'] = tweet_dat['actor']['favoritesCount']
            cur_user_data['friendsCount'] = tweet_dat['actor']['friendsCount']
            cur_user_data['followersCount'] = tweet_dat['actor']['followersCount']

            try:
                cur_user_data['location'] = tweet_dat['actor']['location']
            except:
                cur_user_data['location'] = None

            cur_user_data['n_corpus_instances'] = 1  # this counts RT @, quote tweets, and posts for this person

            user_data[cur_user_id] = cur_user_data

            return (user_data)

    def process_tweet(self, tweet, user_data):

        try:
            cur_user_id = tweet['actor']['id'].split(':')[-1]

        except:
            return None, user_data

        user_dat = self.process_user_data(tweet, user_data)

        # Get tweet data

        cur_tweet_data = {'body': None, 'postedTime': None, 'retweetCount': None, 'favoritesCount': None,
                          'quoted_status_id': None, 'quoted_status_user_id': None, 'quoted_status_body': None,
                          'quoted_status_user_postedTime': None, 'gnip_url_title': None, 'gnip_url_description': None,
                          'is_RT': False, 'RT_body': None, 'RT_user_id': None, 'RT_id': None,
                          'tweet_id': tweet['id'].split(':')[-1], 'user_id': cur_user_id}

        try:

            cur_tweet_data['body'] = tweet['long_object']['body']

        except:

            cur_tweet_data['body'] = tweet['body']

        cur_tweet_data['postedTime'] = parse(tweet['postedTime'])
        cur_tweet_data['retweetCount'] = tweet['retweetCount']
        if "favoritesCount" in tweet["object"]:
            cur_tweet_data['favoritesCount'] = tweet['object']['favoritesCount']
        else:
            cur_tweet_data['favoritesCount'] = tweet['favoritesCount']

        if cur_tweet_data['body'].startswith('RT @'):
            cur_tweet_data['is_RT'] = True
        # if not cur_tweet_data['is_RT'] and cur_tweet_data['favoritesCount']!=0:
        #     print("found a tweet")
        # else:
        #     print("didn't find it")
        try:
            if tweet['object']:
                user_dat = self.process_user_data(tweet['object'], user_data)

                cur_tweet_data['RT_body'] = tweet['object']['body']
                cur_tweet_data['RT_user_id'] = tweet['object']['actor']['id'].split(':')[-1]
                cur_tweet_data['RT_id'] = tweet['object']['id'].split(':')[-1]

        except:
            pass

        try:
            if tweet['twitter_quoted_status']:
                user_dat = self.process_user_data(tweet['twitter_quoted_status'], user_data)

                cur_tweet_data['quoted_status_id'] = tweet['twitter_quoted_status']['id'].split(':')[-1]
                cur_tweet_data['quoted_status_user_id'] = tweet['twitter_quoted_status']['actor']['id'].split(':')[-1]
                cur_tweet_data['quoted_status_body'] = tweet['twitter_quoted_status']['body']
                cur_tweet_data['quoted_status_user_postedTime'] = parse(tweet['twitter_quoted_status']['postedTime'])
        except:
            pass

        try:
            if tweet['gnip']:
                cur_tweet_data['gnip_url_title'] = []
                cur_tweet_data['gnip_url_description'] = []
                for i in tweet['gnip']['urls']:
                    try:
                        cur_tweet_data['gnip_url_title'].append(i['expanded_url_title'])
                        cur_tweet_data['gnip_url_description'].append(i['expanded_url_description'])

                    except KeyError:
                        pass
        except:
            pass

        return (cur_tweet_data, user_data)


def main():
    tweetPrep = TweetPreprocessor("/home/geev/datasets/metoo_oct_nov")
    tweetPrep.push_tweets_to_db()

if __name__ == '__main__':
    main()
