import re
# import nltk
import json
import gzip
import os
import pickle
import unidecode
import pprint
from datetime import datetime
from tqdm import tqdm
from dateutil.parser import parse
from pymongo import MongoClient
import pandas as pd
from names_dataset import NameDataset
from pymongo.errors import BulkWriteError, CursorNotFound
import pymongo

class TweetPreprocessor():
    def __init__(self, tweet_data_path):
        self.tweet_data_path = tweet_data_path
        self.files = os.listdir(self.tweet_data_path)
        self.client = MongoClient()
        self.db_metoo_tweets = self.client["new_metoo"]
        self.metoo_tweets = self.db_metoo_tweets.metoo_tweets
        self.namedb = NameDataset()

    def get_new_client(self):
        self.client = MongoClient()
        self.db_metoo_tweets = self.client["new_metoo"]
        self.metoo_tweets = self.db_metoo_tweets.metoo_tweets

    def pickle_db(self, save_path="data/data.p", query={"is_RT":True}):
        cursor = self.metoo_tweets.find(query)
        list_cur = list(cursor)
        df = pd.DataFrame(list_cur)
        file = open(save_path, 'wb')
        pickle.dump(df, file)
        file.close()

    def push_tweets_to_db(self):
        tweets = self.read_tweets(self.files)
        i=0
        user_data = {}
        for tweet in tqdm(tweets):
            cur_tweet_data, user_data = self.process_tweet(tweet, user_data)
            if cur_tweet_data is not None:
                try:
                    temp = self.metoo_tweets.insert_one(cur_tweet_data)
                except:
                   self.get_new_client()
                   temp = self.metoo_tweets.insert_one(cur_tweet_data)

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

    def validate_name(self, text, cur_targets, target_dic, target_last_names):
        """
            returns a list of targets that have been mentioned in a text by either first name, last name, or handle

                splits the text by " " or "," or "-" then for each target for each token
                1. if it's a mention (starts with @) checks if it matches target handle then target is validated
                2. if it includes either first name or last name of target  then target is validated
                3. if it includes the exclusion list then target is not validated and we move to next target
        """


        split_text = re.split(' |-|,|_', text.lower())
        valid = None
        valid_targets = []

        for cur_target in cur_targets:
            for i, token in enumerate(split_text):
                if cur_target in token:

                    if token.startswith('@') and target_dic[cur_target]['handle'] == token:
                        valid = 1
                    if split_text[i - 1] in target_dic[cur_target]['fn']:
                        valid = 1
                    if split_text[i - 1] in target_last_names:
                        valid = 1
                    if not self.namedb.search_first_name(unidecode.unidecode(split_text[i - 1])):
                        print("here with", split_text[i-1])
                        valid = 1

                    if split_text[i - 1] in target_dic[cur_target]['other_exclude']:
                        valid = 0

            if valid == 1:
                valid_targets.append(cur_target)
        return (valid_targets)

    def check_target(self, targets, label, text, target_mentions):
        if any(substring in text for substring in targets):
            target_mentions.append(label)
        return (target_mentions)

    def check_targets(self, text):
        text = text.lower()

        target_mentions = []

        target_mentions = self.check_target(['metoo'], 'metoo', text, target_mentions)
        target_mentions = self.check_target(['louis ck', 'louis c.k.', 'louisck'], 'louis_ck', text, target_mentions)
        target_mentions = self.check_target(['kirt webster'], 'kirt_webster', text, target_mentions)
        target_mentions = self.check_target(['nassar'], 'nassar', text, target_mentions)
        target_mentions = self.check_target(['landesman'], 'landesman', text, target_mentions)
        target_mentions = self.check_target(['masterson'], 'masterson', text, target_mentions)
        target_mentions = self.check_target(['scott courtney', 'scottcourtney'], 'scott_courtney', text, target_mentions)
        target_mentions = self.check_target(['affleck'], 'affleck', text, target_mentions)
        target_mentions = self.check_target(['scoble'], 'scoble', text, target_mentions)
        target_mentions = self.check_target(['schwahn'], 'schwahn', text, target_mentions)
        target_mentions = self.check_target(['andy dick', 'andydick'], 'andy_dick', text, target_mentions)
        target_mentions = self.check_target(['venit'], 'venit', text, target_mentions)
        target_mentions = self.check_target(['berganza'], 'berganza', text, target_mentions)
        target_mentions = self.check_target(['seagal'], 'seagal', text, target_mentions)
        target_mentions = self.check_target(['goddard'], 'goddard', text, target_mentions)
        target_mentions = self.check_target(['hafford'], 'hafford', text, target_mentions)
        target_mentions = self.check_target(['halperin'], 'halperin', text, target_mentions)
        target_mentions = self.check_target(['smyre'], 'smyre', text, target_mentions)
        target_mentions = self.check_target(['lebsock'], 'lebsock', text, target_mentions)
        target_mentions = self.check_target(['weinstein'], 'weinstein', text, target_mentions)
        target_mentions = self.check_target(['bob weinstein', 'bobweinstein'], 'bob_weinstein', text, target_mentions)
        target_mentions = self.check_target(['ingenito'], 'ingenito', text, target_mentions)
        target_mentions = self.check_target(['spacey'], 'spacey', text, target_mentions)
        target_mentions = self.check_target(['kreisberg'], 'kreisberg', text, target_mentions)
        target_mentions = self.check_target(['hamilton fish', 'hamiltonfish'], 'hamilton_fish', text, target_mentions)
        target_mentions = self.check_target(['grasham'], 'grasham', text, target_mentions)
        target_mentions = self.check_target(['wieseltier'], 'wieseltier', text, target_mentions)
        target_mentions = self.check_target(['knepper'], 'knepper', text, target_mentions)
        target_mentions = self.check_target(['nelly'], 'nelly', text, target_mentions)
        target_mentions = self.check_target(['ratner'], 'ratner', text, target_mentions)
        target_mentions = self.check_target(['ken baker', 'kenbaker'], 'ken_baker', text, target_mentions)
        target_mentions = self.check_target(['toback'], 'toback', text, target_mentions)
        target_mentions = self.check_target(['dreyfuss'], 'dreyfuss', text, target_mentions)
        target_mentions = self.check_target(['piven'], 'piven', text, target_mentions)
        target_mentions = self.check_target(['weiner'], 'weiner', text, target_mentions)
        target_mentions = self.check_target(['takei'], 'takei', text, target_mentions)
        target_mentions = self.check_target(['ethan kath', 'ethankath'], 'ethan_kath', text, target_mentions)
        target_mentions = self.check_target(['lacey'], 'lacey', text, target_mentions)
        target_mentions = self.check_target(['lockhart steele', 'lockhartsteele'], 'lockhart_steele', text, target_mentions)
        target_mentions = self.check_target(['besh'], 'besh', text, target_mentions)
        target_mentions = self.check_target(['bocanegra'], 'bocanegra', text, target_mentions)
        target_mentions = self.check_target(['wenner'], 'wenner', text, target_mentions)
        target_mentions = self.check_target(['hoffman'], 'hoffman', text, target_mentions)
        target_mentions = self.check_target(['caleb jennings', 'calebjennings'], 'caleb_jennings', text, target_mentions)
        target_mentions = self.check_target(['savino'], 'savino', text, target_mentions)
        target_mentions = self.check_target(['david corn', 'davidcorn'], 'david_corn', text, target_mentions)
        target_mentions = self.check_target(['oreskes'], 'oreskes', text, target_mentions)
        target_mentions = self.check_target(['mendoza'], 'mendoza', text, target_mentions)
        target_mentions = self.check_target(['heatherton'], 'heatherton', text, target_mentions)
        target_mentions = self.check_target(['roy price', 'royprice'], 'roy_price', text, target_mentions)
        target_mentions = self.check_target(['sizemore'], 'sizemore', text, target_mentions)
        target_mentions = self.check_target(['tambor'], 'tambor', text, target_mentions)
        target_mentions = self.check_target(['westwick'], 'westwick', text, target_mentions)
        target_mentions = self.check_target(['von trier', 'vontrier', 'larsvontrier'], 'von_trier', text, target_mentions)
        target_mentions = self.check_target(['david marchant', 'davidmarchant'], 'david_marchant', text, target_mentions)
        target_mentions = self.check_target(["howie rubin", "howierubin"], "howie_rubin", text, target_mentions)
        target_mentions = self.check_target(['blaine'], 'blaine', text, target_mentions)
        target_mentions = self.check_target(['russell simmons', 'russellsimmons'], 'russell_simmons', text, target_mentions)
        target_mentions = self.check_target(['roy moore', 'roymoore'], 'moore', text, target_mentions)
        # for target_list in all_targets_list:
        #     self.check_target(['roy moore', 'roymoore'], target_list[0], text, target_mentions)
        return (target_mentions)

    def process_tweet_targets(self, tweet):
        body_target_mentions = None
        quoted_status_target_mentions = None
        RT_target_mentions = None
        gnip_url_title_mentions = None

        if tweet['body']:
            body_target_mentions = self.check_targets(tweet['body'])

        if tweet['quoted_status_body']:
            quoted_status_target_mentions = self.check_targets(tweet['quoted_status_body'])

        #     if tweet['RT_body']:
        #         RT_target_mentions = check_targets(tweet['RT_body'])

        if tweet['gnip_url_title']:
            try:
                gnip_url_title_mentions = [self.check_targets(i) for i in tweet['gnip_url_title']]
                gnip_url_title_mentions = list(set([i for j in gnip_url_title_mentions for i in j]))
            except:
                pass

        tweet['body_target_mentions'] = body_target_mentions
        tweet['quoted_status_target_mentions'] = quoted_status_target_mentions
        tweet['RT_target_mentions'] = RT_target_mentions
        tweet['gnip_url_title_mentions'] = gnip_url_title_mentions

        if body_target_mentions:
            tweet['body_target_mentions_validated'] = self.validate_name(tweet['body'],
                                                                    body_target_mentions,
                                                                    target_dic,
                                                                    target_last_names)

            if len(tweet['body_target_mentions_validated']) > 0:
                tweet['body_target_mentions_validated_true'] = True

        if quoted_status_target_mentions:
            tweet['quoted_status_target_mentions_validated'] = self.validate_name(tweet['quoted_status_body'],
                                                                             quoted_status_target_mentions,
                                                                             target_dic,
                                                                             target_last_names)

            if len(tweet['quoted_status_target_mentions_validated']) > 0:
                tweet['quoted_status_target_mentions_validated_true'] = True

        if gnip_url_title_mentions:
            gnip_url_title_mentions_validated = [self.validate_name(i,
                                                               gnip_url_title_mentions,
                                                               target_dic,
                                                               target_last_names) for i in tweet['gnip_url_title']]

            gnip_url_title_mentions_validated = list(set([i for j in gnip_url_title_mentions_validated for i in j]))

            tweet['gnip_url_title_mentions_validated'] = gnip_url_title_mentions_validated

            if len(tweet['gnip_url_title_mentions_validated']) > 0:
                tweet['gnip_url_title_mentions_validated_true'] = True

        return (tweet)

    def remove_urls(self, text):
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
        return (text)

    def get_clean_text(self,x):

        clean_text = self.remove_urls(x).strip()

        clean_text_toks = []

        drop = True

        for tok in clean_text.split():

            if drop is True:
                if tok.startswith('@'):
                    pass

                if not tok.startswith('@'):
                    drop = False

            if drop is False:
                clean_text_toks.append(tok)

        clean_text = ' '.join(clean_text_toks)

        return (clean_text)


    def add_tweet_or_quoted_tweet(self,tweet):
        tweet['modified_quote_tweet'] = True

        try:
            body_text = self.get_clean_text(tweet["body"])
            clean_tweet = f'<TWEET>: {body_text}'
        except:
            clean_tweet = '<TWEET>:'

        try:

            quoted_status_body_text = self.get_clean_text(tweet['quoted_status_body'])
            clean_tweet = f'{clean_tweet}\n\n<QUOTED TWEET>: {quoted_status_body_text}'

        except:
            pass

        #     try:

        #         gnip_url_titles_texts = [get_clean_text(i) for i in list(set(tweet['gnip_url_title']))]
        #         gnip_url_descriptions_texts = [get_clean_text(i) for i in list(set(tweet['gnip_url_description']))]

        #         for title, desc in zip(gnip_url_titles_texts, gnip_url_descriptions_texts):

        #             ratio = fuzz.ratio(title.lower(),body_text.lower())
        #             #print(ratio)
        #             if ratio > 85:
        #                  tweet['modified_quote_tweet'] = False

        #             clean_tweet = f'{clean_tweet}\n\n<URL TITLE>: {title}\n<URL DESCRIPTION>: {desc}'
        # #         clean_tweet = f'{clean_tweet}\n\n<URL TITLE>: {gnip_url_title_text}'

        #     except:
        #         pass

        tweet['clean_tweet'] = clean_tweet

        return (tweet)

    def unique(self, sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    def mask_tweets(self, tweet):

        fix_bob = re.compile('bob weinstein|bobweinstein', re.IGNORECASE)

        # targ_counter = defaultdict(lambda:0)

        counter = 0

        # for i, d in enumerate(dat):

        clean_tweet = fix_bob.sub('bob', tweet['clean_tweet'])

        # print(d['clean_tweet'])
        targets = tweet['body_target_mentions']

        try:
            targets += tweet['quoted_status_target_mentions']

        except:
            pass

        try:
            targets += tweet['gnip_url_title_mentions']
        except:
            pass

        targets = self.unique(targets)
        initial_targ_counter = 0

        mask_map = {}

        for targ_i, targ in enumerate(targets):

            cur_targ_split = targ.split('_')

            if 'weinstein' in cur_targ_split:
                if 'bob' in cur_targ_split:
                    cur_targ = 'bob_weinstein'
                else:
                    cur_targ = 'harvey_weinstein'

            else:
                cur_targ = cur_targ_split[-1]

            # targ_counter[cur_targ] +=1

            if cur_targ == 'metoo':
                continue

            # print(cur_targ == 'metoo')
            mask_map[f"<TARGET {targ_i + 1}>"] = target_gold_name[cur_targ]

            if initial_targ_counter == 0:

                tweet['clean_tweet_masked'] = targ_dic[cur_targ].sub(f"<TARGET {targ_i + 1}>", clean_tweet)
                initial_targ_counter = 1
            else:

                tweet['clean_tweet_masked'] = targ_dic[cur_targ].sub(f"<TARGET {targ_i + 1}>",
                                                                     tweet['clean_tweet_masked'])

        #         d['clean_tweet_masked'] = re.sub('')

        # tweet['clean_tweet_masked'] = tweet['clean_tweet_masked'].replace('<TARGET> <TARGET>', '<TARGET>')

        tweet['clean_targets_n'] = targ_i + 1

        counter += 1

        tweet['mask_map'] = mask_map

        # dat[i] = d

        #     if counter % 50000 == 0:
        #         print(counter)

        return (tweet)

    def mask_targets_process(self, tweet):
        tweet = self.add_tweet_or_quoted_tweet(tweet)
        tweet = self.mask_tweets(tweet)
        return (tweet)
def main():
    # steps that need to be done:


    targs = pd.read_csv('../../data/transgression_ambiguity_metoo_breaking_stimuli_all_targets.csv').target.values
    # tweetPrep = TweetPreprocessor("/home/geev/datasets/metoo_oct_nov")
    # tweetPrep.push_tweets_to_db()
    print(targs)
if __name__ == '__main__':
    main()
