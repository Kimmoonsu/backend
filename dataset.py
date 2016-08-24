#coding:utf-8

import pandas as pd

train_df = pd.read_pickle("soma_goods_train.df")

train_df.shape
train_df

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

d_list = []
cate_list = []

for each in train_df.iterrows():
    cate = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])
    d_list.append(each[1]['name'])
    # print each[1]['name']
    cate_list.append(cate)
    #print cate
print len(set(cate_list))
print "ori : " +d_list[0]
cate_dict = dict(zip(list(set(cate_list)), range(len(set(cate_list)))))

print cate_dict[u'디지털/가전;네트워크장비;KVM스위치']
print cate_dict[u'패션의류;남성의류;정장']

import re
from konlpy.tag import Twitter
twitter = Twitter()

n_list = []
import nltk
from nltk import pos_tag, word_tokenize
for d in d_list :
    temp = u''+d
    #print "original : " + temp
# bb = re.sub('[=.#()/?:"["$}]', ' ', b)
    kor_sub = re.sub(u"[^가-힣A-Za-z0-9-_]+", " ", temp)
    ngram_sub = re.sub(u"[^가-힣A-Za-z0-9]+", " ", temp)
    # kor_sub = re.sub(u"[^가-힣0-9]+", " ", temp)
    eng_sub = re.sub(u"[^가-힣A-Za-z0-9-_]+", " ", temp)

    temp2 = twitter.pos(u'' + kor_sub, norm=True, stem=True)
    twitter_str = ""
    for t in temp2 :
        twitter_str += t[0] + " "

    texts = nltk.word_tokenize(eng_sub)
    ngram_str = ""
    for text in texts:
        ngram_str += text
    nltk_pos = nltk.pos_tag(texts)
    nltk_str = ""
    for n in nltk_pos:
        nltk_str += n[0] + " "

    # n-gram

    n1_str = zip(*[ngram_sub[i:] for i in range(1)])
    ngramstring_length1 = ""
    for n1 in n1_str:
        ngramstring_length1 += n1[0] + " "

    n2_str = zip(*[ngram_sub[i:] for i in range(2)])
    ngramstring_length2 = ""
    for n2 in n2_str:
        ngramstring_length2 += n2[0] + "" + n2[1] + " "

    total_str = twitter_str +  "" + nltk_str + "" + ngramstring_length1 + "" +ngramstring_length2
    n_list.append(total_str)


# for n in n_list :
#     print n

# bad_chars = ''',./<>?;:'"[{]}\|!@#$%^&*()_+-='''
y_list = []
for each in train_df.iterrows():
    cate = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])
    y_list.append(cate_dict[cate])

x_list = vectorizer.fit_transform(n_list)
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
import numpy as np
svc_param = {'C':np.logspace(-2,0,20)}
gs_svc = GridSearchCV(LinearSVC(loss='l2'),svc_param,cv=5,n_jobs=4)
gs_svc.fit(x_list, y_list)
print gs_svc.best_params_, gs_svc.best_score_
clf = LinearSVC(C=gs_svc.best_params_['C'])
clf.fit(x_list,y_list)
from sklearn.externals import joblib
joblib.dump(clf,'classify.model',compress=3)
joblib.dump(cate_dict,'cate_dict.dat',compress=3)
joblib.dump(vectorizer,'vectorizer.dat',compress=3)
