
# coding: utf-8

# 1) Load company file

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
company = pd.read_csv('crunchbase-companies.csv',encoding='latin-1')


# In[3]:


#company.dtypes
#company.head()
company.category_code.value_counts().plot('barh',fontsize=7)


# In[4]:


software_company = company[company.category_code=='software']


# In[5]:


#company.category_code
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
software_company.shape


# 2) Setup API to grab comments from reddit using company name

# In[6]:


from psaw import PushshiftAPI
api = PushshiftAPI()
import datetime as dt
end_epoch = int(dt.datetime(2013, 10, 1).timestamp())


# pretrained LSMT

# In[7]:


import numpy as np
wordsList = np.load('/Users/xintongchen/LSTM-Sentiment-Analysis/training_data/wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('/Users/xintongchen/LSTM-Sentiment-Analysis/training_data/wordVectors.npy')
print ('Loaded the word vectors!')
import tensorflow as tf
numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('/Users/xintongchen/LSTM-Sentiment-Analysis/models'))

import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix


# In[8]:


#len(inputText[1:250])


# inputText = ""
# inputMatrix = getSentenceMatrix(inputText)
# predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
# if (predictedSentiment[0] > predictedSentiment[1]):
#     sentiment = "Positive"
# else:
#     sentiment = "Negative"

# In[9]:


company_names = software_company.name
company_names = company_names[1:]
reddit = open('company_on_reddit','w')
for c in company_names:
    comments = ""
    print (c)
    gen_company = api.search_comments(q=c,
                                      before=end_epoch,
                                      subreddit='business')
    result_company = list(gen_company)
    num_comments = len(result_company)
    if num_comments > 1:
        for i in range(0,num_comments-1):
            #comments.append(result_company[i].d_['body'])
            comments = ''.join([comments,(result_company[i].d_['body'])])
    elif num_comments == 1:
        comments = result_company[0][3]
    if num_comments == 0:
        inputText =comments
        sentiment = "NA"
    if len(comments) >=250:
        inputText = comments[0:249]
        inputMatrix = getSentenceMatrix(inputText)
        predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
        if (predictedSentiment[0] > predictedSentiment[1]):
            sentiment = "Positive"
        else:
            sentiment = "Negative"
    elif len(comments) < 250  and len(comments)>0:
        inputText = comments
        inputMatrix = getSentenceMatrix(inputText)
        predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
        if (predictedSentiment[0] > predictedSentiment[1]):
            sentiment = "Positive"
        else:
            sentiment = "Negative"
    print (inputText)
    to_write = [c, str(num_comments), str(sentiment)]
    reddit.write("\t".join(to_write)+"\n")
    print (to_write)
    


# In[ ]:


inputMatrix.shape


# In[ ]:


reddit.close()


# In[8]:


software_company[software_company.name=='Anthology Solutions']


# In[14]:


result_company[0][3]

