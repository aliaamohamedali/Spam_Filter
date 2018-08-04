
# coding: utf-8

# In[1]:


import os, re
import email
import nltk
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB 
from sklearn.svm import LinearSVC, SVC 
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from nltk.corpus import stopwords 
from os import path
from collections import Counter


# In[2]:


TRAIN_DATA_PATH = 'Data\Train'
CROSSVAL_DATA_PATH = 'Data\CrossVal'
STOP_WORDS = set(stopwords.words('English'))
DELIMITERS = [',', '.', '!', '?', '/', '&', '-', ':', ';', '@', '"', "'", '#', '*', '+','=', '[', ']', '(', ')', '{', '}', '%', '<', '>']


# In[3]:


VOCAB_SIZE = 3000
vocabulary = []


# In[4]:


def extract_email_body(email_path):
    email_content = email.message_from_file(open(email_path))
    text = None
    if email_content.is_multipart():
        html = None
        for part in email_content.get_payload():
            if part.get_content_charset() is None:
                text = str(part.get_payload())
                continue
            charset = part.get_content_charset()
        
            if part.get_content_type() == 'text/plain':
                text = str(part.get_payload())
            
            if part.get_content_type() == 'text/html':
                html = str(part.get_payload())
            ## Should be indented out               
        if text is not None:
            return text.strip()
        else:
            return html.strip()   
            
    else:
        text  = str(email_content.get_payload())
        return text.strip()


# In[5]:


email_body = extract_email_body('Data\Train\ham\\1011.eml')
email_body = re.sub(r"<.*?>", "", email_body)
print(email_body)


# In[6]:


def process_email(email_body):
    ### Takes full body of email
    ### Returns Dictionary of email words and their counts 
    
    email_words = []
    ## Dictionary to Return
    email_word_counts = None
    
    ## Remove html
    email_body = re.sub(r"<.*?>", "", email_body)
    ## Handle http
    email_body = re.sub(r"(http|https)://[^\s]*", "httpaddr", email_body)
    ## Handle Email Addresses
    email_body = re.sub(r"[^\s]+@[^\s]+", "emailaddr", email_body)
    ## Handle Numbers
    email_body = re.sub(r"\s+[0-9]+\s+", "num", email_body)
    
    words = email_body.strip().split()
    email_words.extend(''.join(w for w in word.lower() if w not in DELIMITERS) for word in words)
    email_word_counts = Counter(email_words)
    
    email_word_counts.pop('', None)
    
    for stop_word in STOP_WORDS: 
        email_word_counts.pop(stop_word, None)
    
    return email_words, email_word_counts


# In[7]:


email_words, email_word_counts = process_email(email_body)
print(email_word_counts)


# In[8]:


def construct_vocabulary(data_directory):
    ### Takes Input path to (training) data 
    ### Return list of vocabulary (Most common N words where N = VOCAB_SIZE)

    all_words = []
    all_word_counts = None
    
    for email_class in os.listdir(data_directory):
        email_class_path = path.join(data_directory, email_class)
        for email in os.listdir(email_class_path):
            email_path = path.join(email_class_path, email)
            email_body = extract_email_body(email_path)
            email_words, email_word_counts = process_email(email_body)
        
            all_words.extend(email_words)
            
    all_word_counts = Counter(all_words) 
    
    all_word_counts.pop('', None)
    
    for stop_word in STOP_WORDS: 
        all_word_counts.pop(stop_word, None)
    
    return np.array(all_word_counts.most_common(VOCAB_SIZE))[:, 0]


# In[9]:


vocabulary = construct_vocabulary(TRAIN_DATA_PATH)
print(vocabulary)


# In[10]:


def extract_email_features(email_path, model = 'svm'):
    ### Takes path to an email file
    ### Returns feature vector of email
    email_body = extract_email_body(email_path)
    _ , email_word_counts = process_email(email_body)
    
    feature_vector = []
    
    ## For SVM and multinomial NB - we add word count to  feature vector 
    if model == 'svm' or model == 'multinomialNB':
        for word in vocabulary:
            feature_vector.extend([email_word_counts.get(word, 0)])
    ## For bernouolli bNB - we add 1 if word in email and 0 otherwise        
    elif model == 'bernoulliNB':
        for word in vocabulary:
            if word in email_word_counts:
                feature_vector.extend([1])
            else:
                feature_vector.extend([0])
    else:
        print('No support for such model')
    return feature_vector            


# In[11]:


feature_vector = extract_email_features('Data\Train\spam\\1.eml')
print(feature_vector)


# In[12]:


def prepare_data_files(data_path, model = 'svm'):
    
    ## Matrix of size (m, VOCAB_SIZE)
    features = []
    ## Vector of size (m)
    labels = []
    
    for email_class in os.listdir(data_path):
        if email_class == r'Ham':
            label = 0
        else:
            label = 1
        emails_dir = path.join(data_path, email_class)
        for email in os.listdir(emails_dir):
            email_path = path.join(emails_dir, email)
            features.append(extract_email_features(email_path, model))
            labels.extend([label])
            
    X = np.array(features)
    y = np.array(labels)
    
#    print(X.shape)
#    print(y.shape)

    return X, y    


# In[13]:


X_train, y_train = prepare_data_files(TRAIN_DATA_PATH)


# In[14]:


print(X_train.shape)
print(y_train.shape)
print(y_train)


# In[15]:


X_cval, y_cval = prepare_data_files(CROSSVAL_DATA_PATH)


# In[16]:


print(X_cval.shape)
print(y_cval.shape)
print(y_cval)


# In[17]:


### Training Support vector Machine
## Set different values of C
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 2000]
train_score = np.zeros(len(C_values))
train_recall = np.zeros(len(C_values))
train_precision = np.zeros(len(C_values))

cv_score = np.zeros(len(C_values))
cv_recall = np.zeros(len(C_values))
cv_precision = np.zeros(len(C_values))

idx = 0
for c_val in C_values:
    svm = SVC(C = c_val)
    svm.fit(X_train, y_train)    
    
    train_score[idx] = svm.score(X_train, y_train)
    train_recall[idx] = metrics.recall_score(y_train, svm.predict(X_train))
    train_precision[idx] = metrics.precision_score(y_train, svm.predict(X_train))
    
    cv_score[idx] = svm.score(X_cval, y_cval)
    cv_recall[idx] = metrics.recall_score(y_cval, svm.predict(X_cval))
    cv_precision[idx] = metrics.precision_score(y_cval, svm.predict(X_cval))
    
    idx += 1


# In[18]:


matrix = np.matrix(np.c_[C_values, train_score, train_recall, train_precision, cv_score, cv_recall, cv_precision])
models = pd.DataFrame(data = matrix, columns = ['C', 'Train Accuracy', 'Train Recall', 'Train Precision', 'CV Accuracy', 'CV Recall', 'CV Precision'])

models.head(n = 9)


# In[19]:


## get model with Precision = 1 and biggest Accuracy
best_model_idx =  models[models['CV Precision']==1]['CV Accuracy'].idxmax()
best_C = C_values[best_model_idx]

models.iloc[best_model_idx, :]

svm = SVC(C = best_C)
svm.fit(X_train, y_train) 


# In[20]:


train_confusion_matrix = confusion_matrix(y_train, svm.predict(X_train))


# In[21]:


print(train_confusion_matrix)


# In[22]:


cv_confusion_matrix = confusion_matrix(y_cval, svm.predict(X_cval))


# In[23]:


print(cv_confusion_matrix)


# In[24]:


### Training MultinomialNB
## Set different values of Alpha
alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 20]
train_score = np.zeros(len(alpha_values))
train_recall = np.zeros(len(alpha_values))
train_precision = np.zeros(len(alpha_values))

cv_score = np.zeros(len(alpha_values))
cv_recall = np.zeros(len(alpha_values))
cv_precision = np.zeros(len(alpha_values))

idx = 0
for alpha_val in alpha_values:
    mNB = MultinomialNB(alpha = alpha_val)
    mNB.fit(X_train, y_train)    
    
    train_score[idx] = mNB.score(X_train, y_train)
    mNB_predictions = mNB.predict(X_train)
    train_recall[idx] = metrics.recall_score(y_train, mNB_predictions)
    train_precision[idx] = metrics.precision_score(y_train, mNB_predictions)
    
    cv_score[idx] = mNB.score(X_cval, y_cval)
    mNB_predictions = mNB.predict(X_cval)
    cv_recall[idx] = metrics.recall_score(y_cval, mNB_predictions)
    cv_precision[idx] = metrics.precision_score(y_cval, mNB_predictions)
    
    idx += 1


# In[25]:


matrix = np.matrix(np.c_[alpha_values, train_score, train_recall, train_precision, cv_score, cv_recall, cv_precision])
models = pd.DataFrame(data = matrix, columns = ['Alpha', 'Train Accuracy', 'Train Recall', 'Train Precision', 'CV Accuracy', 'CV Recall', 'CV Precision'])

models.head(n = 9)


# In[26]:


## Get index of model with max precision
best_model_index = models['CV Precision'].idxmax()
best_alpha = alpha_values[best_model_index]

models.iloc[best_model_index, :]

mNB = MultinomialNB(alpha = best_alpha)
mNB.fit(X_train, y_train)


# In[27]:


X_train_bNB, y_train_bNB = prepare_data_files(TRAIN_DATA_PATH, 'bernoulliNB')
X_cval_bNB, y_cval_bNB = prepare_data_files(CROSSVAL_DATA_PATH, 'bernoulliNB')


# In[28]:


### Training MultinomialNB
## Set different values of Alpha
alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 20]
train_score = np.zeros(len(alpha_values))
train_recall = np.zeros(len(alpha_values))
train_precision = np.zeros(len(alpha_values))

cv_score = np.zeros(len(alpha_values))
cv_recall = np.zeros(len(alpha_values))
cv_precision = np.zeros(len(alpha_values))

idx = 0
for alpha_val in alpha_values:
    bNB = BernoulliNB(alpha = alpha_val)
    bNB.fit(X_train_bNB, y_train_bNB)    
    
    train_score[idx] = bNB.score(X_train_bNB, y_train_bNB)
    bNB_predictions = bNB.predict(X_train_bNB)
    train_recall[idx] = metrics.recall_score(y_train_bNB, bNB_predictions)
    train_precision[idx] = metrics.precision_score(y_train_bNB, bNB_predictions)
    
    cv_score[idx] = bNB.score(X_cval_bNB, y_cval_bNB)
    bNB_predictions = bNB.predict(X_cval_bNB)
    cv_recall[idx] = metrics.recall_score(y_cval_bNB, bNB_predictions)
    cv_precision[idx] = metrics.precision_score(y_cval_bNB, bNB_predictions)
    
    idx += 1


# In[29]:


matrix = np.matrix(np.c_[alpha_values, train_score, train_recall, train_precision, cv_score, cv_recall, cv_precision])
models = pd.DataFrame(data = matrix, columns = ['Alpha', 'Train Accuracy', 'Train Recall', 'Train Precision', 'CV Accuracy', 'CV Recall', 'CV Precision'])

models.head(n = 9)


# In[30]:


## Get index of model with max precision
best_model_index = models['CV Precision'].idxmax()
best_alpha = alpha_values[best_model_index]

models.iloc[best_model_index, :]

bNB = BernoulliNB(alpha = best_alpha)
bNB.fit(X_train, y_train)

