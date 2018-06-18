
# coding: utf-8

# 1) Load company file

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
company = pd.read_csv('crunchbase-companies.csv',encoding='latin-1')


# In[2]:


#company.dtypes
#company.head()
company.category_code.value_counts().plot('barh',fontsize=7)


# In[3]:


software_company = company[company.category_code=='software']


# In[4]:


#company.category_code
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
software_company.shape


# 2) Setup API to grab comments from reddit using company name

# In[5]:


from psaw import PushshiftAPI
api = PushshiftAPI()
import datetime as dt
end_epoch = int(dt.datetime(2013, 10, 1).timestamp())


# In[ ]:


company_names = software_company.name

reddit = open('company_on_reddit2','w')
for c in company_names:
    #comments = []
    comments = ""
    print (c)
    gen_company = api.search_comments(q=c,
                                      before=end_epoch,
                                      subreddit='business')
    result_company = list(gen_company)
    num_comments = len(result_company)
    if num_comments > 1:
        for i in range(0,num_comments-1):
	#    comments.append(result_company[i].d_['body'])
		comments = ' '.join([comments, result_company[i].d_['body']])
    elif num_comments == 1:
        comments = result_company[0][3]
    else:
        num_comments = 0 
    to_write = [c, str(num_comments), str(comments)]
    reddit.write("\t".join(to_write)+"\n")
    print (to_write)
    


# In[ ]:


reddit.close()


# In[8]:


software_company[software_company.name=='Anthology Solutions']


# In[14]:


result_company[0][3]

