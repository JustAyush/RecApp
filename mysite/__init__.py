
import pandas as pd
import numpy as np
from sklearn import preprocessing
from lightfm import LightFM
# import seaborn as sns
# import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix 
from scipy.sparse import coo_matrix 
import scipy
import time
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from lightfm.evaluation import auc_score
import pickle
import re
import pymongo
from random import shuffle

import gensim
import os
import random

from sklearn import preprocessing


import json

import time
from apscheduler.schedulers.background import BackgroundScheduler

import datetime


import torch
import pickle
import torchtext
import spacy


REFRESH_INTERVAL = 60 #seconds
 
scheduler = BackgroundScheduler()
scheduler.start()


init = 'youtube'

myclient=pymongo.MongoClient('mongodb://localhost:27017/')
mydb=myclient['majorProject']
mycol=mydb['bookDataset']


def get_recommendation(userId):
    y, heading = similar_recommendation(model_pickle, user_item_matrix_pickle, userId , user_dikt_pickle,threshold = 7)
    z = json.dumps(y)
    rec_books = json.loads(z)
    return rec_books, heading


def similar_recommendation(model, interaction_matrix, user_id, user_dikt, 
                               threshold = 0,number_rec_items = 15):
    myclient=pymongo.MongoClient('mongodb://localhost:27017/')
    mydb=myclient['majorProject']
    mycol=mydb['bookDataset']
    #Function to produce user recommendations

    # x=mydb['userActivity'].aggregate([{"$match":{"isFifteen":0}},{"$addFields":{"size":{"$size":"$activity"}}},{"$match":{"size":{"$gt":0}}},{"$project":{"user_id":1,"_id":0,"activity":{"book_id":1,"activity":{"net_rating":1}}}},{"$unwind":"$activity"},{"$project":{"user_id":1,"activity":"$activity.book_id","rating":"$activity.activity.net_rating"}}]);
    # y=mydb['userActivity'].aggregate([{"$match":{"isFifteen":0}},{"$addFields":{"size":{"$size":"$activity"}}},{"$match":{"size":{"$gt":0}}},{"$project":{"user_id":1,"_id":0}}]);
    # userlist=list(y)
    # interaction_data=list(x)

    # for i in range (len(userlist)):
    #     mycol.update({"user_id":userlist[i]['user_id']},{"$set":{"isFifteen":1 }})

    # for i in range(len(interaction_data)):
    #     user_item_matrix_pickle[interaction_data[i]['activity']][int(userlist[i]['user_id'])] =int(interaction_data[i]['rating'])

    print(user_id)

    x=mydb['userActivity'].find({"user_id":user_id,"isFifteen":1})
    isFifteen=x.count()
    print('------------------------------------------isFifteen---------------------------')
    
    print(isFifteen)


    if (isFifteen == 1):
        print("-------------------15 books------------------------------")
        n_users, n_items = interaction_matrix.shape
        user_x = user_dikt[user_id]
        scores = pd.Series(model.predict(user_x,np.arange(n_items)))
        scores.index = interaction_matrix.columns
        scores = list(pd.Series(scores.sort_values(ascending=False).index))

        known_items = list(pd.Series(interaction_matrix.loc[user_id,:][interaction_matrix.loc[user_id,:] > threshold].index).sort_values(ascending=False))

        scores = [x for x in scores if x not in known_items]
        # print(len(scores))
        score_list = scores[0:number_rec_items]

        # known_items = list(pd.Series(known_items).apply(lambda x: item_dikt[x]))
        # scores = list(pd.Series(score_list).apply(lambda x: item_dikt[x]))
        scores1 = list(pd.Series(score_list))

        w=mycol.aggregate([{"$match":{"ISBN":{"$in":scores1}}},
                     {"$project":{'_id':0,'ISBN':'$ISBN', 'bookTitle':'$Book-Title','bookAuthor':'$Book-Author','genres':'$genres','imageURL':'$Image-URL','averageRating':'$average_rating','publicationYear':'$publication_year','description':'$description'} }])
        y=list(w)
        heading = "Recommended books"

    else:  
        
        lastCheckedIn = mydb['userActivity'].aggregate([{"$match":{"user_id": user_id}},{"$project":{"activity":{"book_id":1,"activity":{"date_modified":1}}}},{"$unwind":"$activity"},{"$project":{"activity.book_id":1,"date_modified":"$activity.activity.date_modified"}},{"$sort":{"date_modified":-1}}])
        last_check_in = list(lastCheckedIn)

        if(len(last_check_in)!=0):
            s_books = []
            print("last",last_check_in)
            for i in range(len(last_check_in)):
                last_check_in_book_id = last_check_in[i]['activity']['book_id']
                similar_books = getSimilarBooks(user_id, last_check_in_book_id)
                s_books=s_books+similar_books

            s_books=list(set(s_books))

            s_books=random.sample(s_books,15 )
            
            x=mydb['bookDataset'].aggregate([{"$match":{"ISBN":{"$in":s_books}}},{"$project":{'_id':0, 'ISBN':'$ISBN', 'genres': '$genres', 'bookTitle': '$Book-Title', 'bookAuthor': '$Book-Author', 'publicationYear': '$Year-Of-Publication', 'publisher': '$Publisher', 'imageURL': '$Image-URL', 'averageRating': '$average_rating', 'description': '$description', 'publicationYear':'$publication_year'} }])
            y = list(x)
            heading = "Recommended (CBF)"

        else:
            print("--------------------------random-----------------------------------")
            w=mycol.aggregate([{"$match":{"average_rating":{"$gt":4}}},{"$sample":{"size":15}},
                            {"$project":{'_id':0,'ISBN':'$ISBN','bookTitle':'$Book-Title','bookAuthor':'$Book-Author','genres':'$genres','imageURL':'$Image-URL','averageRating':'$average_rating','publicationYear':'$publication_year','description':'$description'} }])
            y = list(w)
            heading = "Random books"

    return y, heading

# name="name1"


# new_name=get_new_name(name)

# get_new_name(name)
# {
#     if(name="name1")
#     {
#         new_name="name2"
#     }
#     else
#     {
#         new_name="name1"
#     }
#     return new_name

# }





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn

model= torch.load("sentiment_model.pt",map_location='cpu')

with open('t_vocab.pickle', 'rb') as handle:
    text_vocab = pickle.load(handle)

import spacy
nlp = spacy.load('en')

def predict_sentiment(model, sentence):
    print("function call")
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    print("type 1 is",type(tokenized))
    indexed = [text_vocab[t] for t in tokenized]
    print("type 2 is",type(indexed))
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

def get_clicks_rating(n):
    return 5*(1-0.5**n)

def get_review_rating(in_text):
    print('Review here ----------------------------------------------', in_text)
    print('in_text type', type(in_text))
    OldValue = predict_sentiment(model, str(in_text))
    print('OldValue', OldValue)
    # OldValue=0.8
    OldMax = 1
    OldMin = 0

    NewMax = 5
    NewMin = 1
    
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    NewValue = round(NewValue)
    print(NewValue)
    
    return NewValue


def get_net_rating(review_rating,rating,clicks_rating):
    review_rating = float(review_rating)
    rating = float(rating)
    clicks_rating = float(clicks_rating)
    weight=0
    weighted_rating=0
    if review_rating!=0:
        weight=weight+1
        weighted_rating=weighted_rating+review_rating
    if rating!=0:
        weight=weight+0.8
        weighted_rating=weighted_rating+rating*0.8
    if clicks_rating!=0:
        weight=weight+0.2
        weighted_rating=weighted_rating+clicks_rating*0.2
    net_rating = weighted_rating/weight
    return net_rating
    # return 5

def get_recommendation(userId):

    with open('model.pickle', 'rb') as handle:
        model_pickle = pickle.load(handle)

    with open('item_dikt.pickle', 'rb') as handle:
        item_dikt_pickle = pickle.load(handle)

    with open('user_item_matrix.pickle', 'rb') as handle:
        user_item_matrix_pickle = pickle.load(handle)

    with open('user_dikt.pickle', 'rb') as handle:
        user_dikt_pickle = pickle.load(handle)

    y = similar_recommendation(model_pickle, user_item_matrix_pickle, userId , user_dikt_pickle,threshold = 7)
    z = json.dumps(y)
    rec_books = json.loads(z)
    return rec_books


# this returns list of user_id from interaction matrix (row name)
# key is user_id value is matrix index
def user_item_dikts(interaction_matrix):
    user_ids = list(interaction_matrix.index)
    user_dikt = {}
    counter = 0 
    for i in user_ids:
        user_dikt[i] = counter
        counter += 1
    return user_dikt



def talkShow():
    print('---------------------------------------James Corden---------------------------------------------')
    myClient=pymongo.MongoClient("localhost:27017")
    mydb=myClient['majorProject']
    mycol=mydb['userActivity']

    # import interaction matrix file
    with open('user_item_matrix.pickle', 'rb') as handle:
        user_item_matrix_pickle = pickle.load(handle)


    # for modifying date
    x=mydb['date_modified'].find({},{"_id":0,"date_modified":1})
    last_date_modified=list(x)[0]['date_modified']
    # print("the date modified is:",last_date_modified)


    # This adds new books at the end of interaction matrix 
    # The last_date_modified needs to be updated here. Right now, the date is harcoded
    x=mydb['bookDataset'].aggregate([{"$match":{"date_added":{"$gt":last_date_modified}}},{"$project":{"_id":0,"ISBN":1}}])
    data=list(x)
    print("the following newly added books are: added")
    for i in data:
        print(i['ISBN'])
        user_item_matrix_pickle[i['ISBN']]=0

    # For isFifteen =1 users and newly modified activity only
    x=mycol.aggregate([{"$match":{"activity.activity.date_modified":{"$gte":last_date_modified}}},{"$unwind":"$activity"},{"$match":{"isFifteen":1}},{"$match":{"activity.activity.date_modified":{"$gte":last_date_modified}}},{"$project":{"_id":0,"activity.activity.net_rating":1,"activity.activity.date_modified":1,"activity.book_id":1,"user_id":1}}])
    data=list(x)
    a=[]
    print("new activity are update now")
    for i in range(len(data)):
        a.append({"user_id":data[i]['user_id'],"activity":data[i]['activity']['book_id'],"rating":data[i]['activity']['activity']['net_rating']})


    # obtain activity of user (isFifteen= 0 and rating>15) on book
    x=mydb['userActivity'].aggregate([{"$match":{"isFifteen":0}},{"$addFields":{"size":{"$size":"$activity"}}},{"$match":{"size":{"$gt":15}}},{"$project":{"user_id":1,"_id":0,"activity":{"book_id":1,"activity":{"net_rating":1}}}},{"$unwind":"$activity"},{"$project":{"user_id":1,"activity":"$activity.book_id","rating":"$activity.activity.net_rating"}}]);
    interaction_data=list(x)
    # obtain the list of new user to update (isFifteen =0 and rating>15)
    y=mydb['userActivity'].aggregate([{"$match":{"isFifteen":0}},{"$addFields":{"size":{"$size":"$activity"}}},{"$match":{"size":{"$gt":15}}},{"$project":{"user_id":1,"_id":0}}]);
    userlist=list(y)

    # add both users
    interaction_data=interaction_data+a


    # For isFifteen=0 users
    # this add new user to interaction matrix and initialize it to zero
    print("newly added useers are:")
    for i in range (len(userlist)):
        # this add new user to interaction matrix and initialize it to zero
        user_item_matrix_pickle.loc[int(userlist[i]['user_id'])]=0
        print(userlist[i]['user_id'])
       
    #     this update isFifteen flag value
        mycol.update({"user_id":userlist[i]['user_id']},{"$set":{"isFifteen":1 }})
        

    # updates all the new ratings
    # this loops throught all the value interaction book and update the values with respective rating
    print("interaction matrix are update now")
    print(interaction_data)
    for i in range(len(interaction_data)):
        print("book no",i)
        user_item_matrix_pickle[interaction_data[i]['activity']][int(interaction_data[i]['user_id'])] =round(2*float(interaction_data[i]['rating'])) 
           
    #for modifying date
    mydb['date_modified'].update({},{"$set":{"date_modified":datetime.datetime.now()}})

    # after interaction matrix is update, this convert pandas dataframe into scipy sparse matrix
    user_item_matrix_sci = scipy.sparse.csr_matrix(user_item_matrix_pickle.values)


    user_dikt = user_item_dikts(user_item_matrix_pickle)

    model_pickle=LightFM(no_components=115,learning_rate=0.027,loss='warp')
    model_pickle.fit(user_item_matrix_sci,epochs=12,num_threads=4)
    print("new data train here")

    with open('user_item_matrix.pickle', 'wb') as handle:
        pickle.dump(user_item_matrix_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("user_item_matrix is saved here")
    with open('model.pickle', 'wb') as handle:
        pickle.dump(model_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("model is saved here")

    with open('user_dikt.pickle', 'wb') as handle:
        pickle.dump(user_dikt, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("user dictionary is saved here")

talkShow()
scheduler.add_job(talkShow, 'interval', seconds = REFRESH_INTERVAL)



# for CBF (doc2vec)

with open('model_doc2vec.pickle','rb') as handle:
    model_doc2vec = pickle.load(handle)
with open('train_corpus.pickle','rb') as handle:
    train_corpus=pickle.load(handle)
with open('label_encoder.pickle','rb') as handle:
    le=pickle.load(handle)


def getSimilarBooks(book_id):
    
    print(book_id)
    books=mydb['bookDataset'].find({"ISBN":book_id})
    book=list(books)

    try:
        doc_id=list(le.transform([book_id]))[0]
        # le.inverse_transform([88])

        sims = model_doc2vec.docvecs.most_similar([model_doc2vec.infer_vector(train_corpus[doc_id].words)], topn=len(model_doc2vec.docvecs))
        
    except:    
        # document=gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line),[i])
        document=gensim.utils.simple_preprocess(book[0]['description'])
        sims = model_doc2vec.docvecs.most_similar([model_doc2vec.infer_vector(document)], topn=len(model_doc2vec.docvecs))
       
    book_array=[]
    for i in range (15):
        book_array.append(str(list(le.inverse_transform([sims[i][0]]))[0]))
    # x=mydb['bookDataset'].find({"ISBN":{"$in":book_array}})
    # x = list(x)
    return book_array



# to fetch book rating and review if present.

def fetchActivity(userId, ISBN):
    import pymongo
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["majorProject"]
    user_activity = mydb['userActivity'].aggregate([{"$match":{"user_id": userId}},{"$unwind":"$activity"},{"$match":{"activity.book_id":ISBN}},{"$project":{"book_id":"$activity.book_id","_id":0,"review":"$activity.activity.review","rating":"$activity.activity.rating","date_modified":"$activity.activity.date_modified"}}])
    user_activity = list(user_activity)

    if(len(user_activity)==0):
        send_data={}
        send_data['review']=''
        send_data['rating']=0
    else:
        if not "review" in user_activity[0]:
            user_activity[0]['review']=''
        if not "rating" in user_activity[0]:
            user_activity[0]['rating']=0
        send_data=user_activity[0]

    return send_data