#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd 
from collections import Counter
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import re  
from collections import defaultdict 
import spacy  
from time import time
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize,sent_tokenize
import seaborn as sns
sns.set_style("darkgrid")

import itertools
from wordcloud import WordCloud
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.layers import Embedding

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# In[19]:


import pandas as pd 
dataframe = pd.read_csv('C:/Users/amala/OneDrive/Desktop/Twitter_Data.csv')
dataframe.head(2)


# In[20]:


dataframe.isnull().sum()


# In[21]:


dataframe = dataframe.dropna().reset_index(drop=True)
dataframe.isnull().sum()


# In[22]:


dataframe['clean_text']= dataframe['clean_text'].str.lower()


# In[23]:


# removing URLs

def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: cleaning_URLs(x))


# In[24]:


# removing numbers 

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: cleaning_numbers(x))


# In[25]:


# fucntion to removes pattern in the input text.

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

# remove twitter handles (@user)

dataframe['clean_text'] = np.vectorize(remove_pattern)(dataframe['clean_text'], "@[\w]*")


# In[26]:


# remove special characters, numbers and punctuations

dataframe['clean_text'] = dataframe['clean_text'].str.replace("[^a-zA-Z#]", " ")
dataframe.head()


# In[27]:


# below code taken from https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)


# In[28]:


brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in dataframe['clean_text'])


# In[45]:


import spacy
from time import time
import pandas as pd

# Download the 'en_core_web_sm' model
spacy.cli.download("en_core_web_sm")

# Load the model
nlp = spacy.load("en_core_web_sm")

t = time()
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

df_clean = pd.DataFrame({'clean': txt})
dataframe.head(5)


# In[46]:


df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
print(df_clean.shape)
df_clean.head(5)


# In[47]:


# How many unique words in the vocabulary?

all_words = " ".join([sentence for sentence in df_clean['clean']])
all_words = all_words.split()

freq_dict = {}
for word in all_words:
    # set the default value to 0
    freq_dict.setdefault(word, 0)
    # increment the value by 1
    freq_dict[word] += 1

voc_freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse = True))
print(len(voc_freq_dict))
hist_plot = dict(itertools.islice(voc_freq_dict.items(), 10))
plt.bar(hist_plot.keys(), hist_plot.values(), width=0.5, color='g')
plt.xticks(rotation=90)
plt.show()


# In[48]:


# Removing stop words using nltk lib

#Tokenization of text
tokenizer=ToktokTokenizer() 

#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

#Removing standard english stopwords like prepositions, adverbs
stop = set(stopwords.words('english'))
print("NLTK stop word lists \n")
print(stop)

#Removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


# In[49]:


df_clean['clean'] = df_clean['clean'].apply(remove_stopwords)
df_clean.head()


# In[50]:


# How many unique words in the vocabulary?

all_words = " ".join([sentence for sentence in df_clean['clean']])
all_words = all_words.split()

freq_dict = {}
for word in all_words:
    # set the default value to 0
    freq_dict.setdefault(word, 0)
    # increment the value by 1
    freq_dict[word] += 1

voc_freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse = True))
print(len(voc_freq_dict))


# In[51]:


# top 10 words with frequency.

hist_plot = dict(itertools.islice(voc_freq_dict.items(), 10))
plt.bar(hist_plot.keys(), hist_plot.values(), width=0.5, color='g')
plt.xticks(rotation=90)
plt.show()


# In[52]:


# visualize the frequent words

all_words = " ".join([sentence for sentence in df_clean['clean']])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[53]:


sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]


# In[54]:


all_sentances = ' '
j = 0
for i in sentences:
    check = ' '.join(i)  
    all_sentances = ' '.join([all_sentances, check])


# In[55]:


# wordcloud freq graph after bigram.

all_sentances = ' '
j = 0
for i in sentences:
    check = ' '.join(i)  
    all_sentances = ' '.join([all_sentances, check])
    
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_sentances)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[56]:


# Count the number of cores in a computer.
cores = multiprocessing.cpu_count() 
print(cores)


# In[57]:


model = Word2Vec(
    sentences = sentences,
    compute_loss=True
)
model.get_latest_training_loss()


# In[58]:


# Here it builds the vocabulary from a sequence of sentences and thus later can be used in the model.

model.build_vocab(sentences, progress_per=10000)
model.corpus_count


# In[59]:


class MetricCallback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self, every=10):
        self.myloss = []
        self.epoch = 0
        self.every = every

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            self.myloss.append(loss)
        else:
            self.myloss.append(loss - self.loss_previous_step)
        if self.epoch % self.every == 0:
            print(f'Loss after epoch {self.epoch}: {self.myloss[-1]}') 
        self.epoch += 1
        self.loss_previous_step = loss


# In[60]:


metric = MetricCallback(every=1)
model = Word2Vec(
    sentences = sentences,
    vector_size=300,
    max_vocab_size = model.corpus_count,
    compute_loss=True,
    callbacks=[metric],
    alpha=0.03,
    min_alpha=0.0007, 
    workers=cores-1
)
plt.plot(metric.myloss)


# In[62]:


metric = MetricCallback(every=1)
model = Word2Vec(
    sentences = sentences,
    vector_size=300,
    max_vocab_size = model.corpus_count,
    compute_loss=True,
    callbacks=[metric],
    alpha=0.03,
    min_alpha=0.0007, 
    workers=cores-1,
    epochs = 10,
)
plt.plot(metric.myloss)


# In[63]:


# the data have lot of political info on India, checkig the positive similarity with word India. 

model.wv.most_similar(positive=['india'])


# In[64]:


# negative similarity with word India. 

model.wv.most_similar(negative=['india'])


# In[65]:


model.wv.most_similar(positive=['bjp'])


# In[66]:


model.wv.most_similar(negative=['bjp'])


# In[67]:


model.wv.most_similar(positive=['narendramodi'])


# In[68]:


model.wv.most_similar(negative=['election'])


# In[69]:


model.wv.most_similar(positive=['election'])


# In[90]:


def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 15 dimensions with PCA
    reduc = PCA(n_components=15).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=5).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))


# In[91]:


# checking Most similar words for bjp vs. 4 Random words:
# red : actual word
# green : random words
# blue :  most similar words
tsnescatterplot(model, 'bjp', ['dog', 'bird', 'bob', 'apu'])


# In[92]:


tsnescatterplot(model, 'india', [i[0] for i in model.wv.most_similar(negative=["india"])])


# In[93]:


# checking Most similar and not similar words for BJP

tsnescatterplot(model, 'narendramodi', [i[0] for i in model.wv.most_similar(negative=["narendramodi"])])


# In[94]:


# checking Most similar and not similar words for congress

tsnescatterplot(model, 'congress', [i[0] for i in model.wv.most_similar(negative=["congress"])])


# In[95]:


# checking Most similar and not similar words for congress

tsnescatterplot(model, 'congress', [i[0] for i in model.wv.most_similar(negative=["congress"])])


# In[96]:


# checking Most similar and not similar words for rahul

tsnescatterplot(model, 'rahul', [i[0] for i in model.wv.most_similar(negative=["rahul"])])


# In[97]:


# Save the model

model.save('tweet-election.w2v')


# In[98]:


model = KeyedVectors.load('tweet-election.w2v')


# In[99]:


model


# In[ ]:




