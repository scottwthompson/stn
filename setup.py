#!/usr/bin/env python


import re
import json
import ijson
import cPickle as pickle
import numpy as np
import Queue

from db import session, User, Business, Tip, userAndRespect
total_t = 0
unigram_features = ['not','good','food','place','like','out','great','very','really','about','go','time','back','service','dont']

buisness_word_frequencies = {}
buisness_social_feature_rating_data = {}
top_70k_users = {}
q = Queue.PriorityQueue(70000)	#70k most respected users. where respect is their total vote counts from other users for: funny, useful, cool
x=0

def load_json(file_name):
    """
        Returns a list containing dictionaries of the parsed JSON.
    """
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f.readlines()]



#num_users = 686556
#top_10% =~ 70000
def find_num_users():
    filename = 'data/yelp_academic_dataset_user.json'
    y = 0
    with open(filename, 'r') as f:
        for line in f:   
	    y = y + 1
	    print(y)

#~1min
def find_top_70k_users():
    filename = 'data/yelp_academic_dataset_user.json'
    y = 0
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)   
	    y = y + 1
	    if y % 10000 == 0:
		print('{0}/70~ complete'.format(y/10000))
	    user_respect = int(data["votes"]["funny"]) + int(data["votes"]["funny"]) + int(data["votes"]["funny"])
	    user_id = data["user_id"]
	    if q.full():
	        first = q.get()
	        if first.count < user_respect:
	            q.put( userAndRespect(user_respect, user_id) )
	        else:
		    q.put(first)
	    else:
	        q.put( userAndRespect(user_respect, user_id) )

    for elem in list(q.queue):
	top_70k_users[elem.id] = elem.count
    


#Run this once should take 15minutes approx
def find_word_frequencies():
    filename = 'data/yelp_academic_dataset_review.json'
    x = 0
    with open(filename, 'r') as f:
        for line in f:
	    x = x + 1
	    if x % 100000 == 0:
		print('{0}/270~ complete'.format(x/10000))
            data = json.loads(line)
	    bid = data["business_id"]
	    user_id = data['user_id']
	    if bid not in buisness_word_frequencies:
	        buisness_word_frequencies[bid] = {}
	        for word in unigram_features:
		    buisness_word_frequencies[bid][word] = 0

	    if bid not in buisness_social_feature_rating_data:
		buisness_social_feature_rating_data[bid] = {}

            words = re.sub(r'([^\s\w]|_)+', '', data['text']).lower().split()
	    for word in words:
	        if word in unigram_features:
	            buisness_word_frequencies[bid][word] += 1
	    if user_id in top_70k_users:
		user_respect = top_70k_users[user_id]
		buisness_social_feature_rating_data[bid][user_id] = [data['stars'], user_respect]

#used to calculate reasonable review count for success weighting of avg stars
def average_num_of_reviews():
    count = 0
    total = 0
    for b in session.query(Business):
	count = count + 1
	total = total +  b.review_count
    print(total/count)
 

#Run this once after populating the buisness_word_frequencies by running find_word_frequencies() (make sure it's done in same session)

#<1min
def import_data_to_sql():
    print "Importing businesses..."
    businesses = load_json('data/yelp_academic_dataset_business.json')
    total_t = 0
    for b in businesses:
        bid = b['business_id']
        latitude = b['latitude']
        longitude = b['longitude']
	stars = b['stars']
	review_count = b['review_count']
	weighted_sucess_rating = (0.5 * stars) + 5 * (1 - 0.5) * (1-np.exp(-review_count/49))
        success = 0
	total_t += weighted_sucess_rating
	if weighted_sucess_rating > 3.75: #calibrated so approx 40% of top rated buisnesses are "successfull" - discussed in report
	    success = 1

	if bid not in buisness_word_frequencies:
	    buisness_word_frequencies[bid] = {}
	    for word in unigram_features:
		buisness_word_frequencies[bid][word] = 0

	social_network_feature = 0.0
	total_r = 0.0

	if bid in buisness_social_feature_rating_data:
	    dict = buisness_social_feature_rating_data[bid]

	    for review in dict:
	        total_r += int(dict[review][1])

	    for review in dict:
	        social_network_feature += (float(dict[review][0])-2.5) * (float(dict[review][1])/total_r)

        session.add(Business(
            bid=bid,
	    stars = stars,
	    review_count = review_count,
            latitude=latitude,
            longitude=longitude,
	    Not = buisness_word_frequencies[bid]['not'],
	    good = buisness_word_frequencies[bid]['good'],
	    food = buisness_word_frequencies[bid]['food'],
	    place = buisness_word_frequencies[bid]['place'],
	    like = buisness_word_frequencies[bid]['like'],
	    out = buisness_word_frequencies[bid]['out'],
	    great = buisness_word_frequencies[bid]['great'],
	    very = buisness_word_frequencies[bid]['very'],
	    really = buisness_word_frequencies[bid]['really'],
	    about = buisness_word_frequencies[bid]['about'],
	    go = buisness_word_frequencies[bid]['go'],
	    time = buisness_word_frequencies[bid]['time'],
	    back = buisness_word_frequencies[bid]['back'],
	    service = buisness_word_frequencies[bid]['service'],
	    dont = buisness_word_frequencies[bid]['dont'],

            social_network_feature = social_network_feature,
	    success = success

        ))
    session.commit()
    print "Done."
    print(total_t)

def setup():
    print("Part 1 : ~ 1 Minute")
    find_top_70k_users()
    print("Part 2 : ~ 15 Minutes")
    find_word_frequencies()
    print("Part 3 : ~ 1 Minute")
    import_data_to_sql()

setup()

