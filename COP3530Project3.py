#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import random


# In[2]:


data = pd.read_csv("dataset.csv")


# In[3]:


data.head()


# In[4]:


nonNumeric = ['song_name', 'artist_name', 'track_id']
dataClean = data.drop(columns=nonNumeric, errors='ignore')
dataClean = dataClean.select_dtypes(include=[np.number])


# In[5]:


scaler = StandardScaler()
dataStandardized = scaler.fit_transform(dataClean)


# In[6]:


# knn
def knnRecommend(userSong, data, k=5):
    similarities = cosine_similarity(data[userSong].reshape(1,-1), data).flatten()
    recommended = np.argsort(-similarities)[1:k+1]
    return recommended


# In[7]:


# hash-based recommendation
class LSH:
    def __init__(self, numHashes, numBands):
        self.buckets = {}
        self.numHashes = numHashes
        self.numBands = numBands
    def fit(self, data):
        rows, features = data.shape
        for i in range(self.numHashes):
            randProjection = np.random.randn(self.numHashes, data.shape[1])
            hashVals = np.sign(np.dot(data, randProjection.T))
            for j in range(self.numBands):
                startID = j * (self.numHashes // self.numBands)
                endID = startID + (self.numHashes // self.numBands)
                bHash = tuple(hashVals[:, startID:endID].flatten())
                if bHash not in self.buckets:
                    self.buckets[bHash] = []
                self.buckets[bHash].append(i)
    def query(self, songID):
        similarSongs = []
        for bucket in self.buckets.values():
            if songID in bucket:
                similarSongs.extend(bucket)
        return list(set(similarSongs) - {songID})


# In[8]:


userSongID = random.randint(0, dataStandardized.shape[0]-1)


# In[9]:


# knn trial
k = 5
knnRecommendations = knnRecommend(userSongID, dataStandardized, k)
print("KNN Recommendations", knnRecommendations)


# In[10]:


# hash-based trial - FIXME
lsh = LSH(numHashes=20, numBands=5)
lsh.fit(dataStandardized)
lshRecommendations = lsh.query(userSongID)
print("Hash-Based Recommendations", lshRecommendations[:k])


# In[11]:


# Naive Bayes
class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.means[cls] = np.mean(X_c, axis=0)
            self.variances[cls] = np.var(X_c, axis=0)
            self.priors[cls] = X_c.shape[0] / X.shape[0]

    def calcLikelihood(self, cls, x):
        mean = self.means[cls]
        variance = self.variances[cls]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

    def calcPosterior(self, x):
        posteriors = {}
        for cls in self.classes:
            prior = self.priors[cls]
            likelihood = np.prod(self.calcLikelihood(cls, x))
            posteriors[cls] = prior * likelihood
        return posteriors

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = self.calcPosterior(x)
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)


# In[12]:


# Train NB
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(dataStandardized)
nb = NaiveBayes()
nb.fit(dataStandardized, labels)


# In[13]:


# Predict NB
predictions = nb.predict(dataStandardized)
print("Predicted Labels:", predictions)


# In[30]:


# Recommend with NB
def recommendNB(userSongID, X, labels, model, k=5):
    userLabel = model.predict(X[userSongID:userSongID+1])[0]
    recommendations = np.where(labels == userLabel)[0]
    return recommendations[:k]

recommendations = recommendNB(userSongID, dataStandardized, labels, nb, k=5)
print("Recommended Songs", recommendations)


# In[ ]:




