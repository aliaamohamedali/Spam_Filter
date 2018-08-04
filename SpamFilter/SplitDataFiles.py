
# coding: utf-8

# In[1]:


import os
import re
import pandas as pd
from os import path
from shutil import copyfile


# In[2]:


DATA_DIRECTORY = 'DataSources\TR'
SPAM_TRAIN_DIRECTORY = 'Data\Train\Spam'
HAM_TRAIN_DIRECTORY = 'Data\Train\Ham'
SPAM_CROSSVAL_DIRECTORY = 'Data\CrossVal\Spam'
HAM_CROSSVAL_DIRECTORY = 'Data\CrossVal\Ham'


# In[3]:


labels = pd.read_csv('labels.csv')
# print(labels)
# spam_ids = labels.loc[labels['Prediction']==0, 'Id']
# print(spam_ids)


# In[6]:


spam_email_count = 0
ham_email_count = 0
SPAM_TRAIN_COUNT = 600 # 0.7 of spam emails
HAM_TRAIN_COUNT = 1200 # 0.7 of ham emails
for file in os.listdir(DATA_DIRECTORY):
    file_id = re.sub(r"\D", "", file)
    src = path.join(DATA_DIRECTORY, file)
    dest_file_name = file_id + '.eml'
    if labels.loc[int(file_id)-1, 'Prediction'] == 0: # If email is spam
        # Copy file to spam directory
        spam_email_count += 1
        if spam_email_count > SPAM_TRAIN_COUNT:
            dest = path.join(SPAM_CROSSVAL_DIRECTORY, dest_file_name)
        else:
            dest = path.join(SPAM_TRAIN_DIRECTORY, dest_file_name)
        copyfile(src, dest)
    else:
        # Copy file to ham directory
        ham_email_count += 1
        if ham_email_count > HAM_TRAIN_COUNT:
            dest = path.join(HAM_CROSSVAL_DIRECTORY, dest_file_name)
        else:
            dest = path.join(HAM_TRAIN_DIRECTORY, dest_file_name)
        copyfile(src, dest)   

