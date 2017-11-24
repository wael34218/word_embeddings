import tensorflow as tf
import re
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display


window_size = 4
word_min_count = 0
skip_gram = 0
embedding_size = 100
epochs = 1
batch_size = 512

'''
TODO:
    - Negative Sampling
    - SkipGram vs CBOW
    - Agparse
    - word_min_count
    - Number Normalization
'''


def clean(line):
    line = re.sub('[^ء-ي0-9 ]+', '', line)
    line = re.sub('[أآإ]', 'ا', line)
    return re.sub('[أإآ]', 'ﺍ', line)


def one_hot(obj, size):
    temp = np.zeros(size * len(obj))
    for i, token in enumerate(obj):
        temp[i*vocab_size + token] = 1
    return temp

sentences = []

print("Read corpus")
word_count = defaultdict(lambda: 0)
with open("corpus_sample.txt") as f:
    for line in f:
        sentence = clean(line).split()
        sentences.append(sentence)
        for word in sentence:
            word_count[word] += 1

print("Tokenize")
tokens = sorted(word_count, key=word_count.get, reverse=True)

tok2int = {}
int2tok = {}
for i, word in enumerate(tokens):
    if word_count[word] > word_min_count:
        tok2int[word] = i
        int2tok[i] = word

vocab_size = len(int2tok)
train_x = []
train_y = []

print("Prepare training data")
for sentence in sentences:
    for start_index in range(len(sentence) - window_size):
        context = sentence[start_index:min(start_index + window_size, len(sentence) - 1)]
        for i in range(len(context)):
            train_x.append(one_hot([tok2int[context[i]]], vocab_size))
            train_y.append(one_hot([tok2int[w] for j, w in enumerate(context) if j != i],
                                   vocab_size))

# Network
print("Train NN")
output_length = vocab_size * (window_size - 1)
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y = tf.placeholder(tf.float32, shape=(None, output_length))

std = (2 / (vocab_size + embedding_size)) ** 0.5
W1 = tf.Variable(tf.random_normal([vocab_size, embedding_size], mean=0, stddev=std))
b1 = tf.Variable(tf.zeros([embedding_size]))
Y1 = tf.add(tf.matmul(x, W1), b1)


std = (2 / (output_length + embedding_size)) ** 0.5
W2 = tf.Variable(tf.random_normal([embedding_size, output_length], mean=0, stddev=std))
b2 = tf.Variable(tf.zeros([output_length]))
pred = tf.nn.softmax(tf.add(tf.matmul(Y1, W2), b2))

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for iteration in range(epochs):
        total_batch = int(len(train_x)/batch_size)
        for b in range(total_batch):
            start = b * batch_size
            end = start + batch_size
            sess.run(optimizer, feed_dict={x: train_x[start:end], y: train_y[start:end]})
            if b % 10 == 0:
                print("    Loss", b, "of", total_batch,
                      sess.run(loss, feed_dict={x: train_x[start:end], y: train_y[start:end]}))
        print("Iter", iteration, "loss:",
              sess.run(loss, feed_dict={x: train_x[start:end], y: train_y[start:end]}))
    print("Tuning Completed!")
    vectors = sess.run(W1 + b1)

print("TSNE")
vectors = TSNE(n_components=2).fit_transform(vectors)

print("Plot")
fig = plt.figure(figsize=(18, 18), dpi=280)
plt.plot()
ax = fig.add_subplot(111)
for i in range(50):
    print(int2tok[i], vectors[i][0], vectors[i][1])
    plt.plot(vectors[i][0], vectors[i][1])
    ax.annotate(get_display(arabic_reshaper.reshape(int2tok[i])), xy=vectors[i], fontsize=8)
plt.show()
