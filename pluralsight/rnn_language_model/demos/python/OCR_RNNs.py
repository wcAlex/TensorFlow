
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
# 
# *next_id = -1 when the current word has ended and the next character is the start of a new word*

# In[86]:

lines[1][:20]


# In[57]:

len(lines)


# In[21]:

def get_features_labels(lines):
    lines = sorted(lines, key=lambda x: int(x[0]))
    data, target = [], []
    
    next_id = -1
    
    word = []
    word_pixels = []

    for line in lines:
         # The index for the next_id column
        next_id = int(line[2])

        # An image for a single character, reshaped
        pixels = np.array([int(x) for x in line[6:134]])
        pixels = pixels.reshape((16, 8))
        
        # Word pixels are a list of 16x8 images which form a single word
        word_pixels.append(pixels)
        
        # Append together the characters which make up a word
        word.append(line[1])
        
        if next_id == -1:
            data.append(word_pixels)
            target.append(word)

            word = []
            word_pixels = []


    return data, target


# In[22]:

data, target = get_features_labels(lines)


# #### The total number of words in our dataset

# In[24]:

len(data), len(target)


# #### All words lengths should be the same
# 
# * Get every word to be the same length as the longest word in our dataset
# * Pad the words with empty characters

# In[25]:

def pad_features_labels(data, target):    
    max_length = max(len(x) for x in target)
    
    # Set up image representations for the empty string (all pixels set to 0)
    padding = np.zeros((16, 8))

    # Pad the image data with the empty string images
    data = [x + ([padding] * (max_length - len(x))) for x in data]
    
    # Pad the words with empty string characters
    target = [x + ([''] * (max_length - len(x))) for x in target]
    
    return np.array(data), np.array(target)


# In[26]:

padded_data, padded_target = pad_features_labels(data, target)


# In[27]:

padded_target[:10]


# In[29]:

padded_target[200:210]


# #### The length of each sequence
# 
# We've padded all words so that their lengths are all equal to the length of the longest word

# In[51]:

word_length = len(padded_target[0])


# In[52]:

word_length


# #### Tensor shape
# 
# * 6877 words
# * Each word padded to have 14 characters
# * Each character represented by 16x8 image

# In[30]:

padded_data.shape


# In[31]:

padded_data.shape[:2] + (-1,)


# In[32]:

reshaped_data = padded_data.reshape(padded_data.shape[:2] + (-1,))


# #### Reshape the data so the image is a 1-D array of pixels

# In[33]:

reshaped_data.shape


# #### Tensor shape
# 
# * 6877 words
# * Each an array with 14 characters (padded with empty strings as needed)

# In[34]:

padded_target.shape


# #### One-hot representation
# 
# * Each character has a feature vector of 26 (only lower case characters)

# In[35]:

padded_target.shape + (26,)


# In[36]:

one_hot_target = np.zeros(padded_target.shape + (26,))


# ### Numpy.ndenumerate is a way to get all indices needed to access elements of a matrix
# <pre>
# a = numpy.array([[1,2],[3,4],[5,6]])
# for (x,y), value in numpy.ndenumerate(a):
#   print x,y 
# </pre>
#  
# 0 0 <br>
# 0 1 <br>
# 1 0 <br>
# 1 1 <br>
# 2 0 <br>
# 2 1 <br>

# In[37]:

for index, letter in np.ndenumerate(padded_target):
    if letter:
        one_hot_target[index][ord(letter) - ord('a')] = 1


# #### One-hot representation of the letter 'o'
# 
# * The letter 'o' represented by a 1 at the 14th index 
# * Index positions start at 0

# In[42]:

one_hot_target[0][0]


# In[43]:

shuffled_indices = np.random.permutation(len(reshaped_data))

shuffled_data = reshaped_data[shuffled_indices]
shuffled_target = one_hot_target[shuffled_indices]


# #### Split into training and test data

# In[44]:

split = int(0.66 * len(shuffled_data))

train_data = shuffled_data[:split]
train_target = shuffled_target[:split]

test_data = shuffled_data[split:]
test_target = shuffled_target[split:]


# In[45]:

train_data.shape


# In[46]:

_, num_steps, num_inputs = train_data.shape


# In[47]:

train_target.shape


# In[48]:

num_classes = train_target.shape[2]


# In[49]:

tf.reset_default_graph()


# In[50]:

X = tf.placeholder(tf.float64, [None, num_steps, num_inputs])

y = tf.placeholder(tf.float64, [None, num_steps, num_classes])


# #### Sequence length calculation
# 
# *['How', 'are', 'you', 'doing'] ==> [14, 14, 14, 14] ==> [3, 3, 3, 5]*
#  
#  The actual length of each word (without the padding) in the input batch

# In[89]:

# All real characters will have a max value of 1, padded characters will be represented by 0s
used = tf.sign(tf.reduce_max(tf.abs(X), reduction_indices=2))

# Sum up the number of real characters for each word
length = tf.reduce_sum(used, reduction_indices=1)
sequence_length = tf.cast(length, tf.int32)


# In[90]:

sequence_length


# #### RNN for training and prediction

# In[91]:

num_neurons = 300


# In[92]:

cell = tf.nn.rnn_cell.GRUCell(num_neurons)


# #### *sequence_length* is the length of the valid input for each batch
# 
# Included to improve accuracy and not for performance

# In[93]:

output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float64, sequence_length=sequence_length)


# In[94]:

output.shape


# #### Shared softmax layer

# In[95]:

weight = tf.Variable(tf.truncated_normal([num_neurons, num_classes], stddev=0.01, dtype=tf.float64))


# In[96]:

bias = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float64))


# In[97]:

flattened_output = tf.reshape(output, [-1, num_neurons])


# In[98]:

flattened_output


# In[99]:

logits = tf.matmul(flattened_output, weight) + bias


# In[100]:

logits_reshaped = tf.reshape(logits, [-1, num_steps, num_classes])


# #### Cost calculation

# In[101]:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)


# In[102]:

loss = tf.reduce_mean(cross_entropy)


# #### Error calculation
# 
# * For every word calculate how many of the characters we predicted correctly
# * Use the mask to not consider (leave out) the padded characters on which our prediction was wrong
# * Find the fraction of each word where we made mistakes in our character prediction
# * Find the average fraction of each word that were mistakes

# In[103]:

mistakes = tf.not_equal(
            tf.argmax(y, 2), tf.argmax(logits_reshaped, 2))
mistakes = tf.cast(mistakes, tf.float64)
mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
mistakes *= mask


# In[104]:

mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
mistakes /= tf.cast(sequence_length, tf.float64)


# In[105]:

error = tf.reduce_mean(mistakes)


# #### Optimizer

# In[106]:

optimizer = tf.train.RMSPropOptimizer(0.002)


# In[107]:

gradient = optimizer.compute_gradients(loss)


# In[108]:

optimize = optimizer.apply_gradients(gradient)


# In[109]:

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


# In[112]:

batch_size = 20
batches = batched(train_data, train_target, batch_size)


# In[113]:

epochs = 5


# In[115]:

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



