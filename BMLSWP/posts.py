import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import scipy as sp
import sys
import nltk.stem

def dist_raw(v1, v2):
    #Calculate the vector distance not on the raw vector but on normazlized ones.
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

#Stemmer to extend the vectorizer
english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
    """ Extending CountVectorizer class to include stemming """
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

class StemmedTfidfVectorizer(TfidfVectorizer):
    """ Extends the TfidfVectorizer to include stemming """
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

#min_df is the minimum document frequency. All features with frequency below this will be dropped. stop_words are not included.
vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
vectorizer_tdidf = StemmedTfidfVectorizer(min_df=1, stop_words='english')
posts = [open(os.path.join('toy_data', f)).read() for f in os.listdir('toy_data') ]
X_train = vectorizer.fit_transform(posts)
X_train_tfidf = vectorizer_tdidf.fit_transform(posts)

num_samples, num_features = X_train.shape
num_samples_tfidf, num_features_tfidf = X_train_tfidf.shape

print (vectorizer.get_feature_names())
print ('%d, %d' % (num_samples, num_features))

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
new_post_vec_tfidf = vectorizer_tdidf.transform([new_post])


best_doc = None
best_dist = sys.maxsize
best_i = None
for i in range(0, num_samples):

    post = posts[i]

    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_raw(post_vec, new_post_vec)

    print ("=== Post %i with dist=%.2f: %s" % (i, d, post))
    if d < best_dist:
        best_dist = d
        best_i = i

print ("best post is %i with dist=%.2f"%(best_i, best_dist))

best_doc_tfidf = None
best_dist_tfidf = sys.maxsize
best_j= None

for j in range(0, num_samples_tfidf):

    post_td = posts[j]

    if post_td == new_post:
        continue
    post_td_vec = X_train_tfidf.getrow(j)
    d_td = dist_raw(post_td_vec, new_post_vec_tfidf)

    print ("=== Post %i with dist=%.2f: %s" % (j, d_td, post))
    if d_td < best_dist_tfidf:
        best_dist_tfidf = d_td
        best_j = j

print ("best post is %i with dist=%.2f"%(best_j, best_dist_tfidf))
