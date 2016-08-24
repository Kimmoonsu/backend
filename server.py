
# coding: utf-8

# In[1]:

from sklearn.externals import joblib
from IPython.display import clear_output


# In[2]:


clf = joblib.load('classify.model')
cate_dict = joblib.load('cate_dict.dat')
vectorizer = joblib.load('vectorizer.dat')


# In[3]:

joblib.dump(clf,'n_classify.model')


# In[4]:

joblib.dump(cate_dict,'n_cate_dict.dat')
joblib.dump(vectorizer,'n_vectorizer.dat')


# In[5]:

cate_id_name_dict = dict(map(lambda (k,v):(v,k),cate_dict.items()))


# In[ ]:

pred = clf.predict(vectorizer.transform(['[신한카드5%할인][서우한복] 아동한복 여자아동 금나래 (분홍)']))[0]
print cate_id_name_dict[pred]


# In[ ]:

from bottle import route, run, template,request,get, post
import re
from konlpy.tag import Twitter
twitter = Twitter()
from nltk import pos_tag, word_tokenize
import nltk
import  time
from threading import  Condition

_CONDITION = Condition()
test=0
@route('/classify')
def classify():
    global test
    test = test + 1
    
    print "classify called: " + str(test)
    
    if test % 100 == 0:
        clear_output()
    
    img = request.GET.get('img','')
    name = request.GET.get('name', '')
    print "input : " + name

        
# nltk import

    kor_sub = re.sub(u"[^가-힣A-Za-z0-9-_]+", " ", name.decode("utf-8"))
    ngram_sub = re.sub(u"[^가-힣A-Za-z0-9]+", " ", name.decode("utf-8"))
    eng_sub = re.sub(u"[^가-힣A-Za-z0-9-_]+", " ", name.decode("utf-8"))
    print "kor = " + kor_sub
    print "eng : " + eng_sub
    temp2 = twitter.pos(kor_sub, norm=True, stem=True)
    twitter_str = ""
    for t in temp2 :
        twitter_str += t[0] + " "

    texts = nltk.word_tokenize(eng_sub)
    
    ngram_str = ""
    for text in texts:
        ngram_str += text
    
    temp = nltk.pos_tag(texts)
    nltk_str = ""
    for n in temp :
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

    total_str = twitter_str + "" + nltk_str + "" + ngramstring_length1 + "" + ngramstring_length2
    print "total : " + total_str

    pred = clf.predict(vectorizer.transform([total_str]))[0]
    return {'cate':cate_id_name_dict[pred]}
    

run(host='0.0.0.0', port=5557)


# In[ ]:



