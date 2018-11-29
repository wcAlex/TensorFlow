
# coding: utf-8

# ### Character prediction using RNNs

# In[1]:

get_ipython().system(u'pip install --upgrade numpy')
get_ipython().system(u'pip install --upgrade tensorflow')
get_ipython().system(u'pip install --upgrade bs4')


# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:

import os
import requests
import random


# In[3]:

import numpy as np
import tensorflow as tf


# In[4]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt


# In[5]:

from six.moves import urllib
from bs4 import BeautifulSoup


# In[6]:

print(np.__version__)
print(tf.__version__)


# ### Download technical paper summaries
# 
# * Papers are on machine learning, neural networks
# * The downloaded data is 100MB so will take a long time to write out
# * Roughly 100K papers with these categories and keywords present

# In[7]:

BASE_PATH = 'http://export.arxiv.org/api/query'
CATEGORIES = [
    'Machine Learning',
    'Neural and Evolutionary Computing',
    'Optimization'
]
KEYWORDS = [
    'neural',
    'network',
    'deep'    
]


# In[8]:

def build_url(amount, offset):
    categories = ' OR '.join('cat:' + x for x in CATEGORIES)
    keywords = ' OR '.join('all:' + x for x in KEYWORDS)

    url = BASE_PATH
    url += '?search_query=(({}) AND ({}))'.format(categories, keywords)
    url += '&max_results={}&offset={}'.format(amount, offset)
    
    return url


# In[9]:

build_url(0, 0)


# #### Beautiful soup and the *lxml* parser
# 
# Very clear directions on what Beautiful Soup is and how to get it on your local machine
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# 
# Install Beautiful Soup
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-beautiful-soup
# 
# Get the **lxml** parser
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-a-parser

# In[10]:

def get_count():
    url = build_url(0, 0)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    
    count = int(soup.find('opensearch:totalresults').string)
    print(count, 'papers found')
    
    return count


# In[11]:

num_papers = get_count()


# In[12]:

PAGE_SIZE = 100

def fetch_page(amount, offset):
    url = build_url(amount, offset)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    
    for entry in soup.findAll('entry'):
        text = entry.find('summary').text
        text = text.strip().replace('\n', ' ')
        yield text

def fetch_all():
    for offset in range(0, num_papers, PAGE_SIZE):
        print('Fetch papers {}/{}'.format(offset + PAGE_SIZE, num_papers))
        
        for page in fetch_page(PAGE_SIZE, offset):
            yield page


# #### Huge amount of data
# 
# This takes a while to run. Alternatively you can just download the **arxiv_abstracts.txt** from here: https://goo.gl/QoZH4Y and place it in your current working directory

# In[13]:

DOWNLOADED_FILENAME = 'arxiv_abstracts.txt'

def download_data():
    if not os.path.isfile(DOWNLOADED_FILENAME):
        with open(DOWNLOADED_FILENAME, 'w') as file_:
            for abstract in fetch_all():
                file_.write(abstract + '\n')
    with open(DOWNLOADED_FILENAME) as file_:
        data = file_.readlines()
        
    return data    


# In[14]:

data = download_data()


# In[15]:

len(data)


# #### The number of time steps used when we train our data
# 
# All of the sentences in our abstracts are divided into windows of 50 characters. Our RNN will have 50 layers

# In[16]:

MAX_SEQUENCE_LENGTH = 50
BATCH_SIZE = 100


# In[17]:

VOCABULARY =         " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"         "\\^_abcdefghijklmnopqrstuvwxyz{|}"


# #### Assign a unique number to represent each character
# 
# The number representation of the character is used as an index in the one-hot representation of characters

# In[18]:

lookup = {x: i for i, x in enumerate(VOCABULARY)}


# In[19]:

sample_lookup = random.sample(lookup.items(), 10)
sample_lookup


# #### One-hot representation of each character
# 
# * Every window of characters is of MAX_SEQUENCE_LENGTH (50)
# * Each character is represented in one-hot notation
# * Every feature vector has length equal to number of characters in the vocabulary

# In[20]:

def one_hot(batch, sequence_length = MAX_SEQUENCE_LENGTH):
    one_hot_batch = np.zeros((len(batch), sequence_length, len(VOCABULARY)))

    # Iterate through every line of text in a batch
    for index, line in enumerate(batch):
        line = [x for x in line if x in lookup]
        assert 2 <= len(line) <= MAX_SEQUENCE_LENGTH
        
        # Iterate through every character in a line
        for offset, character in enumerate(line):
            # Code is the index of the character in the vocabulary
            code = lookup[character]
 
            one_hot_batch[index, offset, code] = 1
    
    return one_hot_batch


# #### Sliding window over every line
# 
# * Generate batches of characters for training data
# * Start the sliding window at index 0 for every line
# * Slide the window over till the last character in the line is included
# * Have a stride of MAX_SEQUENCE_LENGTH // 2 for every window move

# In[23]:

def next_batch():
    windows = []
    for line in data:
        for i in range(0, len(line) - MAX_SEQUENCE_LENGTH + 1, MAX_SEQUENCE_LENGTH // 2):
            windows.append(line[i: i + MAX_SEQUENCE_LENGTH])

    # All text at this point are in the form of windows of MAX_SEQUENCE_LENGTH characters
    assert all(len(x) == len(windows[0]) for x in windows)

#     print('Number of windows: ', len(windows))
#     print('Length of one window: ', len(windows[0]))

    while True:
        random.shuffle(windows)
        for i in range(0, len(windows), BATCH_SIZE):
            batch = windows[i: i + BATCH_SIZE]
            yield one_hot(batch)


# In[22]:

test_batch = None
for batch in next_batch():
    test_batch = batch
    print(batch.shape)
    break


# #### Set up the inputs to the RNN
# 
# * One batch of MAX_SEQUENCE_LENGTH characters is the input sequence
# * The training X and the target y should be constructed from this input
# * St is the input and St+1 is the target
# * Slice the sequence to get X, **X has the last frame cut away**
# * Slice the sequence to get the corresponding y, **y has the first frame cut away**

# In[24]:

tf.reset_default_graph()


# In[25]:

sequence = tf.placeholder(tf.float32, [None, MAX_SEQUENCE_LENGTH, len(VOCABULARY)])


# In[26]:

X = tf.slice(sequence, (0, 0, 0), (-1, MAX_SEQUENCE_LENGTH - 1, -1))


# In[27]:

y = tf.slice(sequence, (0, 1, 0), (-1, -1, -1))


# In[28]:

X.shape


# In[29]:

y.shape


# #### Sequence length calculation
# 
# The sequence length here will **be the same for all our inputs** because they have been generated using the sliding window.
# 
# We've sliced away either the first frame (for the labels) or the last frame (for the input) so the sequence length will be *MAX_SEQUENCE_LENGTH - 1*

# In[30]:

def get_mask(target):
    mask = tf.reduce_max(tf.abs(target), reduction_indices=2)
    return mask

def get_sequence_length(target):
    mask = get_mask(target)
    sequence_length = tf.reduce_sum(mask, reduction_indices=1)
    
    return sequence_length


# #### RNN for training and prediction

# In[31]:

num_neurons = 200
cell_layers = 2

num_steps = MAX_SEQUENCE_LENGTH - 1
num_classes = len(VOCABULARY)


# In[32]:

sequence_length = get_sequence_length(y)


# ### MultiRNNCell
# 
# Used to stack multiple RNN cells and have them behave as a single cell
# 
# **MultiRNNCell([lstm] * 5)** will build a 5 layer LSTM stack where each layer shares the same parameters
# 
# **MultiRNNCell([LSTMCell(...) for _ in range(5)])** will give 5 layers where each layer has its own parameters

# In[33]:

def build_rnn(data, num_steps, sequence_length, initial=None):
    cell = tf.nn.rnn_cell.GRUCell(num_neurons)

    multi_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.GRUCell(num_neurons) for _ in range(cell_layers)])

    output, state = tf.nn.dynamic_rnn(
        inputs=data,
        cell=multi_cell,
        dtype=tf.float32,
        initial_state=initial,
        sequence_length=sequence_length)

    # Shared softmax layer across all RNN cells
    weight = tf.Variable(tf.truncated_normal([num_neurons, num_classes], stddev=0.01))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    flattened_output = tf.reshape(output, [-1, num_neurons])

    prediction = tf.nn.softmax(tf.matmul(flattened_output, weight) + bias)
    prediction = tf.reshape(prediction, [-1, num_steps, num_classes])

    return prediction, state


# #### Prediction and the last recurrent activation
# 
# In the training phase of this RNN we only use the *prediction* output. The second output, the *last recurrent activation* is ignored. In the prediction phase later on, we will use the last recurrent activation to generate sequences more effectively

# In[34]:

prediction, _ = build_rnn(X, num_steps, sequence_length)


# #### Cost calculation
# 
# Basic cross entropy loss calculated manually on our prediction output from the RNN. This is used in place of the *tf.nn.softmax_cross_entropy_with_logits* library function.

# In[35]:

mask = get_mask(y)

prediction = tf.clip_by_value(prediction, 1e-10, 1.0)

cross_entropy = y * tf.log(prediction)
cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)

cross_entropy *= mask


# In[36]:

length = tf.reduce_sum(sequence_length, 0)

cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) / length
cross_entropy = tf.reduce_mean(cross_entropy)


# #### Calculate the log probability

# In[37]:

logprob = tf.multiply(prediction, y)
logprob = tf.reduce_max(logprob, reduction_indices=2)
logprob = tf.log(tf.clip_by_value(logprob, 1e-10, 1.0)) / tf.log(2.0)


# In[38]:

logprob *= mask

length = tf.reduce_sum(sequence_length, 0)

logprob = tf.reduce_sum(logprob, reduction_indices=1) / length
logprob = tf.reduce_mean(logprob)


# #### Optimizer

# In[39]:

optimizer = tf.train.RMSPropOptimizer(0.002)

gradient = optimizer.compute_gradients(cross_entropy)

optimize = optimizer.apply_gradients(gradient)


# In[45]:

num_epochs = 10
epoch_size = 50

logprob_evals = []


# In[46]:

checkpoint_dir = './sample_checkpoint_output'


# In[47]:

with tf.Session() as sess:
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        for _ in range(epoch_size):
            batch = next(next_batch())
            
            logprob_eval, _ = sess.run((logprob, optimize), {sequence: batch})
            
            logprob_evals.append(logprob_eval)
            
        saver.save(sess, os.path.join(checkpoint_dir, 'char_pred'), epoch)    
        
        perplexity = 2 ** -(sum(logprob_evals[-epoch_size:]) /
                            epoch_size)
        print('Epoch {:2d} perplexity {:5.4f}'.format(epoch, perplexity))


# In[ ]:




# In[ ]:




# In[69]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



