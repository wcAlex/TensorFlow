
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


# In[64]:

URL_PATH = 'http://ai.stanford.edu/~btaskar/ocr/letter.data.gz'
DOWNLOADED_FILENAME = 'letter.data.gz'

def download_data():
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(URL_PATH, DOWNLOADED_FILENAME)
    
    print('Found and verified file from this path: ', URL_PATH)
    print('Downloaded file: ', DOWNLOADED_FILENAME)


# In[65]:

download_data()


# In[66]:

def read_lines():
    with gzip.open(DOWNLOADED_FILENAME, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        lines = list(reader)

        return lines


# In[67]:

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

# In[68]:

lines[0][:8]


# In[69]:

len(lines)


# In[70]:

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


# In[71]:

data, target = get_features_labels(lines)


# In[72]:

def pad_features_labels(data, target):    
    max_length = max(len(x) for x in target)
    padding = np.zeros((16, 8))

    data = [x + ([padding] * (max_length - len(x))) for x in data]
    target = [x + ([''] * (max_length - len(x))) for x in target]
    
    return np.array(data), np.array(target)


# In[73]:

padded_data, padded_target = pad_features_labels(data, target)


# In[74]:

padded_target[:10]


# #### The length of each sequence
# 
# We've padded all words so that their lengths are all equal to the length of the longest word

# In[75]:

sequence_length = len(padded_target[0])


# In[76]:

sequence_length


# In[77]:

padded_data.shape


# In[78]:

padded_data.shape[:2] + (-1,)


# In[79]:

reshaped_data = padded_data.reshape(padded_data.shape[:2] + (-1,))


# In[80]:

reshaped_data.shape


# In[81]:

padded_target.shape


# In[82]:

padded_target.shape + (26,)


# In[83]:

one_hot_target = np.zeros(padded_target.shape + (26,))


# In[84]:

for index, letter in np.ndenumerate(padded_target):
    if letter:
        one_hot_target[index][ord(letter) - ord('a')] = 1


# #### One-hot representation of the letter 'o'
# 
# * The letter 'o' represented by a 1 at the 14th index 
# * Index positions start at 0

# In[85]:

one_hot_target[0][0]


# In[86]:

shuffled_indices = np.random.permutation(len(reshaped_data))

shuffled_data = reshaped_data[shuffled_indices]
shuffled_target = one_hot_target[shuffled_indices]


# In[87]:

split = int(0.66 * len(shuffled_data))

train_data = shuffled_data[:split]
train_target = shuffled_target[:split]

test_data = shuffled_data[split:]
test_target = shuffled_target[split:]


# In[88]:

train_data.shape


# In[89]:

_, num_steps, num_inputs = train_data.shape


# In[90]:

train_target.shape


# In[91]:

num_classes = train_target.shape[2]


# In[92]:

tf.reset_default_graph()


# In[93]:

X = tf.placeholder(tf.float64, [None, num_steps, num_inputs])

y = tf.placeholder(tf.float64, [None, num_steps, num_classes])


# #### Sequence length calculation

# In[94]:

used = tf.sign(tf.reduce_max(tf.abs(X), reduction_indices=2))

length = tf.reduce_sum(used, reduction_indices=1)
sequence_length = tf.cast(length, tf.int64)


# In[95]:

sequence_length


# #### RNN for training and prediction

# In[96]:

num_neurons = 300


# In[97]:

output, _ = tf.nn.bidirectional_dynamic_rnn(tf.nn.rnn_cell.GRUCell(num_neurons), 
                                            tf.nn.rnn_cell.GRUCell(num_neurons),
                                            X,
                                            dtype=tf.float64, sequence_length=sequence_length)


# In[98]:

output


# In[99]:

output = tf.concat([output[0], output[1]], axis=2)


# In[100]:

output.shape


# #### Shared softmax layer

# In[101]:

weight = tf.Variable(tf.truncated_normal([num_neurons * 2, num_classes], stddev=0.01, dtype=tf.float64))


# In[102]:

bias = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float64))


# In[103]:

flattened_output = tf.reshape(output, [-1, num_neurons * 2])


# In[104]:

flattened_output


# In[105]:

logits = tf.matmul(flattened_output, weight) + bias


# In[106]:

logits_reshaped = tf.reshape(logits, [-1, num_steps, num_classes])


# #### Cost calculation

# In[107]:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)


# In[108]:

loss = tf.reduce_mean(cross_entropy)


# #### Error calculation

# In[109]:

mistakes = tf.not_equal(
            tf.argmax(y, 2), tf.argmax(logits_reshaped, 2))
mistakes = tf.cast(mistakes, tf.float64)
mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
mistakes *= mask


# In[110]:

mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
mistakes /= tf.cast(sequence_length, tf.float64)


# In[111]:

error = tf.reduce_mean(mistakes)


# #### Optimizer

# In[112]:

optimizer = tf.train.RMSPropOptimizer(0.002)


# In[113]:

gradient = optimizer.compute_gradients(loss)


# In[114]:

optimize = optimizer.apply_gradients(gradient)


# In[115]:

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


# In[116]:

batch_size = 10
batches = batched(train_data, train_target, batch_size)


# In[117]:

epochs = 5


# In[118]:

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


# In[119]:

batch_size = 20
batches = batched(train_data, train_target, batch_size)


# In[120]:

epochs = 5


# In[121]:

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



