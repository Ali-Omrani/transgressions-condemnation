import re
# import nltk
import datetime
import json
import gzip
import os
import pickle
import unidecode
import pprint
from tqdm import tqdm
import numpy as np
from dateutil.parser import parse
from pymongo import MongoClient
import pandas as pd
from names_dataset import NameDataset
# from whatthelang import WhatTheLang

from pymongo.errors import BulkWriteError, CursorNotFound
import pymongo

class TweetPreprocessor():
    def __init__(self, tweet_data_path, all_targets_list, target_dic):
        self.tweet_data_path = tweet_data_path
        self.files = os.listdir(self.tweet_data_path)
        self.client = MongoClient()
        self.db_metoo_tweets = self.client["new_metoo"]
        self.metoo_tweets = self.db_metoo_tweets.metoo_tweets
        self.namedb = NameDataset()
        self.all_targets_list = all_targets_list
        self.target_dic = target_dic # ln, fn, handle
        self.set_target_regex_dic()
        self.target_last_names = [self.target_dic[k]['ln'] for k in self.target_dic.keys()]
        self.target_gold_name =  {}
        for k in self.targ_dic.keys():
            self.target_gold_name[k] = self.targ_dic[k].pattern.split('|')[0].title()
        self.build_target_article_dates()
        # self.wtl = WhatTheLang()


    def build_target_article_dates(self, vox_data_path='../../data/vox_accusation_data.csv', window_days = 7):
        """
        sets self.target_date
        builds a dictionary of the acceptable date range for each target
        """
        self.target_date = {}

        for k in self.targ_dic.keys():
            self.target_date[k] = {}
            self.target_date[k]['name'] = self.targ_dic[k].pattern.split('|')[0].title().lower()

        vox_data = pd.read_csv(vox_data_path)
        vox_data['public_datetime'] = pd.to_datetime(vox_data.public_date, format='%B %d, %Y')
        vox_data['name_string'] = vox_data.name.str.replace(' ', '_')

        for k in self.target_date.keys():
            time = vox_data[vox_data.name == self.target_date[k]['name']].public_datetime.values

            time = (time - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')
            try:
                time = datetime.datetime.utcfromtimestamp(time[0])
                self.target_date[k]['min_date'] = time
                self.target_date[k]['max_date'] = time + datetime.timedelta(days=window_days)
            except:
                pass
                print("failed to build date for {}".format(k))
                # print(k)

        self.target_date['ck']['min_date'] = datetime.datetime.strptime('November 9, 2017', '%B %d, %Y')
        self.target_date['ck']['max_date'] = datetime.datetime.strptime('November 9, 2017',
                                                                   '%B %d, %Y') + datetime.timedelta(days=window_days)

    def check_date(self, target, postedTime):
        """
        returns True/False
        Checks if the posted time lies within acceptable window for the target
        """
        try:
            if postedTime >= self.target_date[target]['min_date'] and postedTime <= self.target_date[target]['max_date']:
                return (True)
            else:
                return (False)
        except:
            print("exception in check_date for {}".format(target))
            return (None)

    def check_target_dates(self, tweet):
        """
        Checks if the tweet is within time window of first articles of all targets mentioned in the tweet
        """
        targets = self.get_targets(tweet)
        time_check = {}
        for target in targets:
            if target == 'metoo':
                continue
            else:

                if 'weinstein' in target:
                    if target == 'weinstein':
                        target2 = 'harvey_weinstein'
                    else:
                        target2 = target
                else:
                    target2 = target.split('_')[-1]

                #             elif target == 'roy_price':
                #                 target2 = 'price'

                #             elif target == 'louis_ck':
                #                 target2 = 'ck'

                #             elif target == 'kirt_webster':
                #                 target2 = 'webster'

                #             else:
                #                 target2 = target

                time_check[target] = {}
                valid = self.check_date(target2, tweet['postedTime'])

                if valid is None:
                    return (None)

                time_check[target]['valid'] = valid
                time_check[target]['public_date'] = self.target_date[target2]['min_date']

        tweet['time_check'] = time_check
        return (tweet)

    def get_targets(self, tweet):
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
        return (targets)

    def set_target_regex_dic(self):
        print("building target regex")
        self.targ_dic = {}
        for target in self.all_targets_list:
            if len(target)==2:
                target = target[0] # to choose the right element from the list
            try:
                split_targ = target.split('_')
                k = split_targ[-1]
                if 'harvey' in split_targ:
                    k = 'harvey_weinstein'

                if 'bob' in split_targ and 'weinstein' in split_targ:
                    k = 'bob_weinstein'
                    self.targ_dic[k] = re.compile('|'.join(['bob weinstein', 'bobweinstein', 'bob']), re.IGNORECASE)

                self.targ_dic[k] = re.compile('|'.join([' '.join(split_targ)] + split_targ[1:] + [''.join(split_targ)]),
                                         re.IGNORECASE)
            except:
                pass

        self.targ_dic['ck'] = re.compile('|'.join(['louis ck', 'louis c.k.', 'louisck', 'louisc.k.']), re.IGNORECASE)
        self.targ_dic['moore'] = re.compile('|'.join(['roy moore', 'roymoore', 'moore']), re.IGNORECASE)
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
                        # print("here with", split_text[i-1])
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
        #todo: fix
        # target_mentions = self.check_target(['metoo'], 'metoo', text, target_mentions)
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
        #todo: fix
        # target_mentions = self.check_target(['nelly'], 'nelly', text, target_mentions)
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
                                                                    self.target_dic,
                                                                    self.target_last_names)

            if len(tweet['body_target_mentions_validated']) > 0:
                tweet['body_target_mentions_validated_true'] = True

        if quoted_status_target_mentions:
            tweet['quoted_status_target_mentions_validated'] = self.validate_name(tweet['quoted_status_body'],
                                                                             quoted_status_target_mentions,
                                                                             self.target_dic,
                                                                             self.target_last_names)

            if len(tweet['quoted_status_target_mentions_validated']) > 0:
                tweet['quoted_status_target_mentions_validated_true'] = True

        if gnip_url_title_mentions:
            gnip_url_title_mentions_validated = [self.validate_name(i,
                                                               gnip_url_title_mentions,
                                                               self.target_dic,
                                                               self.target_last_names) for i in tweet['gnip_url_title']]

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
        #print(targets)
        targ_i = -1
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
            mask_map[f"<TARGET {targ_i + 1}>"] = self.target_gold_name[cur_targ]

            if initial_targ_counter == 0:

                tweet['clean_tweet_masked'] = self.targ_dic[cur_targ].sub(f"<TARGET {targ_i + 1}>", clean_tweet)
                initial_targ_counter = 1
            else:

                tweet['clean_tweet_masked'] = self.targ_dic[cur_targ].sub(f"<TARGET {targ_i + 1}>",
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

    def mask_all_db(self):
        print("masking all DB")
        counter = 0

        # for document in tqdm(self.metoo_tweets.find({'is_RT': True,
        #                                    '$or': [{'body_target_mentions_validated_true': True},
        #                                            {'quoted_status_target_mentions_validated_true': True},
        #                                            {'gnip_url_title_mentions_validated_true': True},
        #                                            {'body_target_mentions_validated_true': True}],
        #                                    'lang_pred': 'en',
        #                                    'lang_pred_prob': {'$gte': .3}})):
        for document in tqdm(self.metoo_tweets.find({'is_RT': True})):

            document = self.mask_targets_process(document)

            self.update_tweet_in_db(document)



            # counter += 1
            #
            # if counter % 10000 == 0:
            #     print(counter)

    def add_target_mentions_to_db(self):
        print("adding target mentions to DB")
        counter = 0

        for document in tqdm(self.metoo_tweets.find({'is_RT': True})):
            #Todo: fix lang prediction
            # lang_pred = self.wtl.pred_prob(document['body'])
            lang_pred = [[]]
            if len(lang_pred[0]) > 0:

                document['lang_pred'] = lang_pred[0][0][0]
                document['lang_pred_prob'] = lang_pred[0][0][1]

            else:
                document['lang_pred'] = 'unknown'
                document['lang_pred_prob'] = 0

            document = self.process_tweet_targets(document)

            self.update_tweet_in_db(document)

            # counter += 1
            #
            # if counter % 10000 == 0:
            #     print(counter)


    def add_target_date_validation_to_db(self):
        print("adding target date check to db")
        for document in tqdm(self.metoo_tweets.find({'is_RT': True})):
                                           # '$or': [{'body_target_mentions_validated_true': True},
                                           #         {'quoted_status_target_mentions_validated_true': True}],
                                           # 'lang_pred': 'en',
                                           # 'lang_pred_prob': {'$gte': .3},
                                           # 'time_check': {'$exists': False}}):

            document = self.check_target_dates(document)

            if document is None:
                print("ERROR in add TARGET dates!")
                break

            self.update_tweet_in_db(document)

    def update_tweet_in_db(self, document):

        try:
            self.metoo_tweets.update_one(
                {'_id': document['_id']},
                {'$set': document}
            )

        except:
            self.get_new_client()
            self.metoo_tweets.update_one(
                {'_id': document['_id']},
                {'$set': document}
            )



def get_all_targ_list(targs):
    all_targets_list = []
    for t_name in targs:

        if not isinstance(t_name, str):
            continue
        if ("_" not in t_name):
            print("bad {}".format(t_name))
            all_targets_list.append(t_name)
            continue
            #     print(t_name)
        if len(t_name.split("_")) != 2:
            print("too long or too short", t_name)
            all_targets_list.append([t_name, " ".join(t_name.split("_"))])
            continue

        first, last = t_name.split("_")
        all_targets_list.append([t_name, " ".join([first, last])])
    return all_targets_list
def main():
    # steps that need to be done:
    # clean the tweets - get target mentions - validate target mentions
    # Add language Prediciton
    # Adding time check


    targs = pd.read_csv('../../data/transgression_ambiguity_metoo_breaking_stimuli_all_targets.csv').target.values
    all_targets_list = get_all_targ_list(targs)
    with open("../../data/target_dic.json", "r") as f:
        target_dic = json.load(f)
    print(target_dic)
    tweetPrep = TweetPreprocessor("/home/geev/datasets/metoo_oct_nov", all_targets_list, target_dic)
    # tweetPrep.push_tweets_to_db()
    # tweetPrep.add_target_mentions_to_db()
    # tweetPrep.mask_all_db()
    # tweetPrep.add_target_date_validation_to_db()
if __name__ == '__main__':
    main()
