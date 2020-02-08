import json
from twitter import Twitter, OAuth
from itertools import filterfalse
from itertools import Counter

# Configuration File
data = open("Code/config.json")
config = json.load(data)

# Configuration Setup
oauth = OAuth(
    config["access_token"],
    config["access_token_secret"],
    config["consumer_key"],
    config["consumer_secret"]
)


# Configure the OAuth using the Credentials provided
tobj = Twitter(auth=oauth, mode)

# fetch the Tweets and query accordingly, filtered using links
try:
    iterator = tobj.search.tweets(q='#WheelofTime', lang='en', count=50)
except:
    print("ERROR", iterator)


# list which has DICTIONARY for the Tweet JSON
tweets_q = iterator['statuses']
filtered_rt = filterfalse(lambda x: x['text'].startswith("RT"), tweets_q)
filtered_follow = filter(lambda x: x['user']['followers_count'] > 10)

# list of the users who have already tweeted, so as to fetch tweets from different user everytime
users_tweeted = []

# Tweet Count
i = 1

# For every tweet that is fetched, get only relevant tweets
for tweet in filtered_follow:
    if ( tweet['user']['screen_name'] not in users_tweeted):
        print(i, ' '.join(tweet['text'].split("\n")).encode(encoding='utf-8'))
        users_tweeted.append(tweet['user']['screen_name'])
        i += 1
