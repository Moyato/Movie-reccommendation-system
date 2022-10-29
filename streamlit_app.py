import numpy as np
import pandas as pd
import time
import streamlit as st

url="https://raw.githubusercontent.com/alooperalta/movieRecommendation/master/content/ml-latest-small/movies.csv"
movies=pd.read_csv(url)
movies.head()

st.title('Movie Recommendation System')

movie_title=movies['title']
Genres = movies['genres'].str.get_dummies(sep='|')
movies = pd.concat([movies, Genres], axis=1)
movies.drop(['genres','(no genres listed)'],axis=1,inplace=True)
movies.head()

# Load the saved model
# import pickle
# filename = 'recommender.pkl'
# pickle_in = open(filename, 'rb')
# recommender= pickle.load(pickle_in)
# recommender.fit_predict(X)

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import random

X=movies.drop('title',axis=1)
n_clusters=170
recommender=KMeans(n_clusters=n_clusters,random_state=1)
labels=recommender.fit_predict(X)


rndi=dict()
for i in range(n_clusters):
    rndi[i]=[]
for i in range(len(labels)):
    rndi[labels[i]].append(movie_title[i])


def getRecom(x):
    if(x!="Select an Option"):
        ind=movie_title[movie_title==x].index[0]
        choice=random.sample(rndi[labels[ind]],5)
        st.write('\nFinding Recommendations Similar to', x,'...')
        time.sleep(2)
        st.write('\nTop Matches for',x, 'are:\n' )
        for i in choice:
            if (i==x):
                st.write(random.sample(rndi[labels[ind]],1)[0]) 
            else:
                st.write(i)
        st.write("")

def model():        
        o = movie_title
        option = st.selectbox('Select your favourite movie',o)
        getRecom(option)


model()



