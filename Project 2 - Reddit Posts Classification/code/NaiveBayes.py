import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import string
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('reddit_train.csv')

# let's convert are subreddits to numbers
enc = LabelEncoder()
enc.fit(data['subreddits'])
data['subreddits'] = enc.transform(data['subreddits'])

train, test = train_test_split(data, shuffle=False, train_size=0.90, test_size=0.10) #, test_size=0.9)  # let's split the data 75 percent training, 25 for testing
train = train.reset_index()
test = test.reset_index()

# lets get our probablities for each of the words in each of the topics
subreddits = train['subreddits'].unique().tolist()  # getting the unique subreddits

# let's make a list of 20 dictionaries to store our probabilites of the words
theta_j_k = [{} for i in range(0,20)]
theta_k = [1 for i in range(0,20)]  # the number of occurences of the topic
words = {}  # store all the words in that topic in this dictionary
word_count_total = {}  # store the total number of words 
tot_num_words_topic = {}  # store the total number of words for that subreddit

# remove punctuation from each word
table = str.maketrans('', '', string.punctuation)

for topic in subreddits:
    subreddit = train.loc[train.subreddits == topic]
    subreddit = subreddit.reset_index()
    word_set = set()  # store all distinct words in that topic in this list
    tot_num_words_topic[topic] = 2.0
    
    dict = {}  # this dictionary will count the number of occurences of a word in the subreddit

    for index, row in subreddit.iterrows():
        tokens = row['comments'].split()
        tokens = [w.translate(table).lower() for w in row['comments'].split()]

        tot_num_words_topic[topic] += len(tokens)
        for token in tokens:
            if "http" in token or len(token) < 3: 
                continue
            word_set.add(token)
            
            if token in dict:
                dict[token] += 1
                word_count_total[token] += 1

            else:
                dict[token] = 1 + 1  # second 1 for smoothing
                word_count_total[token] = 1.0 + 2.0  # add 2 for smoothing

    words[topic] = word_set
    theta_k[topic] = 1.0 #this is part of laplace smoothing 
    theta_j_k[topic] = dict
   

# calculate P(Y) = P(class)
for topic in train['subreddits'].tolist():
    theta_k[topic] += 1.0

for prob in theta_k:
    prob = prob / (len(subreddits) + 2.0)

# Now we can try testing our model
predictions = []

for index, row in test.iterrows():
    Max = -1.000  # let's store the highest probability we have seen here
    maxTopic = -1  # let's store the topic with the highest probability here
    tokens = row['comments'].split()
    tokens = [t.translate(table).lower() for t in tokens]

    for topic in subreddits:
        prob = theta_k[topic]
        for token in tokens:
            if len(token) < 3 or "http" in token:
                continue
            if token not in words[topic]:
                prob = prob * 1 / tot_num_words_topic[topic]
            else:
                prob = prob * theta_j_k[topic][token]/word_count_total[token]

#         for feature in words[topic]:
#             if feature not in tokens:
#                 try:
#                     prob = prob * (1 - theta_j_k[topic][feature]/word_count_total[token])
#                 except:
#                     continue

        # we need to find which topic had the highest probability for the comment
        if prob > Max:
            Max = prob
            maxTopic = topic

    predictions.append(maxTopic)  # set the prediction of the comment to the topic with the highest probability

# Now let's get our accuracy
realVals = test['subreddits'].values.tolist()
count = 0  # counting the number of correct predictions
for i in range(0, len(realVals)):
    if realVals[i] == predictions[i]:
        count += 1
