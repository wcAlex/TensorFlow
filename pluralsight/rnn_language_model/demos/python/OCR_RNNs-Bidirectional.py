
# coding: utf-8

# ### Optical character recognition using RNNs

# In[1]:

get_ipython().system(u'pip install --upgrade numpy')
get_ipython().system(u'pip install --upgrade tensorflow')


# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[3]:

import os
import gzip
import csv


# In[4]:

import numpy as np
import tensorflow as tf


# In[5]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt


# In[6]:

from six.moves import urllib


# In[7]:

print(np.__version__)
print(tf.__version__)


# In[8]:

URL_PATH = 'http://ai.stanford.edu/~btaskar/ocr/letter.data.gz'
DOWNLOADED_FILENAME = 'letter.data.gz'

def download_data():
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(URL_PATH, DOWNLOADED_FILENAME)
    
    print('Found and verified file from this path: ', URL_PATH)
    print('Downloaded file: ', DOWNLOADED_FILENAME)


# In[9]:

download_data()


# In[10]:

def read_lines():
    with gzip.open(DOWNLOADED_FILENAME, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        lines = list(reader)

        return lines


# In[11]:

lines = read_lines()


# ### Format of every line
# 
# * id
# * letter
# * next_id
# * word_id
# * position
# * fold
# * 16x8 columns of pixel values

# In[12]:

lines[0][:8]


# In[13]:

len(lines)


# In[14]:

def get_features_labels(lines):
    lines = sorted(lines, key=lambda x: int(x[0]))
    data, target = [], []
    
    next_id = -1
    
    word = []
    word_pixels = []

    for line in lines:
        next_id = int(line[2]) # The index for the next_id column

        pixels = np.array([int(x) for x in line[6:134]])
        pixels = pixels.reshape((16, 8))
        
        word_pixels.append(pixels)
        word.append(line[1])
        
        if next_id == -1:
            data.append(word_pixels)
            target.append(word)

            word = []
            word_pixels = []


    return data, target


# In[15]:

data, target = get_features_labels(lines)


# In[16]:

def pad_features_labels(data, target):    
    max_length = max(len(x) for x in target)
    padding = np.zeros((16, 8))

    data = [x + ([padding] * (max_length - len(x))) for x in data]
    target = [x + ([''] * (max_length - len(x))) for x in target]
    
    return np.array(data), np.array(target)


# In[17]:

padded_data, padded_target = pad_features_labels(data, target)


# In[18]:

padded_target[:10]


# #### The length of each sequence
# 
# We've padded all words so that their lengths are all equal to the length of the longest word

# In[19]:

sequence_length = len(padded_target[0])


# In[20]:

sequence_length


# In[21]:

padded_data.shape


# In[22]:

padded_data.shape[:2] + (-1,)


# In[23]:

reshaped_data = padded_data.reshape(padded_data.shape[:2] + (-1,))


# In[24]:

reshaped_data.shape


# In[25]:

padded_target.shape


# In[26]:

padded_target.shape + (26,)


# In[27]:

one_hot_target = np.zeros(padded_target.shape + (26,))


# In[28]:

for index, letter in np.ndenumerate(padded_target):
    if letter:
        one_hot_target[index][ord(letter) - ord('a')] = 1


# #### One-hot representation of the letter 'o'
# 
# * The letter 'o' represented by a 1 at the 14th index 
# * Index positions start at 0

# In[29]:

one_hot_target[0][0]


# In[30]:

shuffled_indices = np.random.permutation(len(reshaped_data))

shuffled_data = reshaped_data[shuffled_indices]
shuffled_target = one_hot_target[shuffled_indices]


# In[31]:

split = int(0.66 * len(shuffled_data))

train_data = shuffled_data[:split]
train_target = shuffled_target[:split]

test_data = shuffled_data[split:]
test_target = shuffled_target[split:]


# In[32]:

train_data.shape


# In[33]:

_, num_steps, num_inputs = train_data.shape


# In[34]:

train_target.shape


# In[35]:

num_classes = train_target.shape[2]


# In[36]:

tf.reset_default_graph()


# In[37]:

X = tf.placeholder(tf.float64, [None, num_steps, num_inputs])

y = tf.placeholder(tf.float64, [None, num_steps, num_classes])


# #### Sequence length calculation

# In[38]:

used = tf.sign(tf.reduce_max(tf.abs(X), reduction_indices=2))

length = tf.reduce_sum(used, reduction_indices=1)
sequence_length = tf.cast(length, tf.int64)


# In[39]:

sequence_length


# #### RNN for training and prediction

# In[40]:

num_neurons = 300


# #### Forward RNN to feed in each word in the right order
# 
# Make sure you specify a scope for each RNN so you can initialize multiple RNNs in the same graph (the default scope is *'rnn'* which will clash across the two RNNs we set up)

# In[41]:

forward, _ = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(num_neurons), X,
                               dtype=tf.float64, sequence_length=sequence_length,
                               scope='rnn-forward')


# #### Reverse the characters in each word and feed in to another forward RNN
# 
# * Reverse the 1st dimension i.e the characters
# * Note that only the actual sequence length of the characters are reversed, the padding is not reversed

# In[42]:

X_reversed = tf.reverse_sequence(X, sequence_length, seq_dim=1)

backward, _ = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(num_neurons), X_reversed,
                               dtype=tf.float64, sequence_length=sequence_length,
                               scope='rnn-backward')


# #### Get output back in the original order

# In[43]:

backward = tf.reverse_sequence(backward, sequence_length, seq_dim=1)


# In[44]:

backward, forward


# In[45]:

output = tf.concat([forward, backward], axis=2)


# In[46]:

output.shape


# #### Shared softmax layer

# In[47]:

weight = tf.Variable(tf.truncated_normal([num_neurons * 2, num_classes], stddev=0.01, dtype=tf.float64))


# In[48]:

bias = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float64))


# In[49]:

flattened_output = tf.reshape(output, [-1, num_neurons * 2])


# In[50]:

flattened_output


# In[51]:

logits = tf.matmul(flattened_output, weight) + bias


# In[52]:

logits_reshaped = tf.reshape(logits, [-1, num_steps, num_classes])


# #### Cost calculation

# In[53]:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)


# In[54]:

loss = tf.reduce_mean(cross_entropy)


# #### Error calculation

# In[55]:

mistakes = tf.not_equal(
            tf.argmax(y, 2), tf.argmax(logits_reshaped, 2))
mistakes = tf.cast(mistakes, tf.float64)

mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
mistakes *= mask


# In[56]:

mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
mistakes /= tf.cast(sequence_length, tf.float64)


# In[57]:

error = tf.reduce_mean(mistakes)


# #### Optimizer

# In[58]:

optimizer = tf.train.RMSPropOptimizer(0.002)


# In[59]:

gradient = optimizer.compute_gradients(loss)


# In[60]:

optimize = optimizer.apply_gradients(gradient)


# In[61]:

def batched(data, target, batch_size):
    epoch = 0
    offset = 0
    while True:
        old_offset = offset
        offset = (offset + batch_size) % (target.shape[0] - batch_size)

        # Offset wrapped around to the beginning so new epoch
        if offset < old_offset:
            # New epoch, need to shuffle data
            shuffled_indices = np.random.permutation(len(data))
            
            data = data[shuffled_indices]
            target = target[shuffled_indices]

            epoch += 1

        batch_data = data[offset:(offset + batch_size), :]
        
        batch_target = target[offset:(offset + batch_size), :]

        yield batch_data, batch_target, epoch


# In[62]:

batch_size = 10
batches = batched(train_data, train_target, batch_size)


# In[63]:

epochs = 5


# In[64]:

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    for index, batch in enumerate(batches):
        batch_data = batch[0]
        batch_target = batch[1]
    
        epoch = batch[2]

        if epoch >= epochs:
            break
        
        feed = {X: batch_data, y: batch_target}
        train_error, _ = sess.run([error, optimize], feed)
        
        print('{}: {:3.6f}%'.format(index + 1, 100 * train_error))

    test_feed = {X: test_data, y: test_target}
    test_error, _ = sess.run([error, optimize], test_feed)
    
    print('Test error: {:3.6f}%'.format(100 * test_error))


# In[ ]:



