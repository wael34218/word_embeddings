import tensorflow as tf
import re
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from sklearn.utils import shuffle


window_size = 4
word_min_count = 6
skip_gram = 0
embedding_size = 75
epochs = 1
batch_size = 512

'''
TODO:
    - Negative Sampling
    - SkipGram vs CBOW
    - Agparse
'''


def clean(line):
    line = re.sub('[^ء-ي0-9 ]+', '', line)
    line = re.sub('[أآإ]', 'ا', line)
    return re.sub('[أإآ]', 'ﺍ', line)


def one_hot(obj, size):
    temp = np.zeros(size * len(obj))
    for i, token in enumerate(obj):
        if token > -1:
            temp[i*vocab_size + token] = 1
    return temp

sentences = []

print("Read corpus")
word_count = defaultdict(lambda: 0)
with open("corpus.txt") as f:
    for line in f:
        sentence = clean(line).split()
        sentences.append(sentence)
        for word in sentence:
            word_count[word] += 1

print("Tokenize")
tokens = sorted(word_count, key=word_count.get, reverse=True)

tok2int = {"PAD": -1}
int2tok = {-1: "PAD"}
for i, word in enumerate(tokens):
    if word_count[word] > word_min_count:
        tok2int[word] = i
        int2tok[i] = word


vocab_size = len(int2tok)
print("Total Vocabs:", vocab_size)

train_x = []
train_y = []

print("Prepare training data")
for sentence in sentences:
    sentence = [w for w in sentence if word_count[w] > word_min_count]
    for word_index in range(len(sentence)):
        context = ["PAD"] * max((window_size - word_index), 0)
        context += sentence[max(word_index - window_size, 0):word_index]
        context += sentence[word_index + 1:min(word_index + window_size + 1, len(sentence))]
        context += ["PAD"] * (window_size * 2 - len(context))
        word = sentence[word_index]
        train_x.append(one_hot([tok2int[word]], vocab_size))
        train_y.append(one_hot([tok2int[w] for w in context], vocab_size))

train_x, train_y = shuffle(train_x, train_y)

# Network
print("Train NN")
output_length = vocab_size * (window_size * 2)
x = tf.placeholder(tf.float64, shape=(None, vocab_size))
y = tf.placeholder(tf.float64, shape=(None, output_length))

stddev = (2 / (vocab_size + embedding_size)) ** 0.5
W1 = tf.Variable(tf.random_normal([vocab_size, embedding_size], stddev=stddev, dtype=tf.float64))
b1 = tf.Variable(tf.zeros([embedding_size], dtype=tf.float64))
Y1 = tf.add(tf.matmul(x, W1), b1)

stddev = (2 / (output_length + embedding_size)) ** 0.5
W2 = tf.Variable(tf.random_normal([embedding_size, output_length], stddev=stddev, dtype=tf.float64))
b2 = tf.Variable(tf.zeros([output_length], dtype=tf.float64))
pred = tf.nn.softmax(tf.add(tf.matmul(Y1, W2), b2))

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(train_x)/batch_size)
    print("Total batches:", total_batch)
    for iteration in range(epochs):
        total_loss = 0
        for b in range(total_batch):
            start = b * batch_size
            end = start + batch_size
            loss_value, _ = sess.run([loss, optimizer],
                                     feed_dict={x: train_x[start:end], y: train_y[start:end]})
            total_loss += loss_value / batch_size
            if (b) % 10 == 0 and b > 0:
                print("   batch loss", b, "is", loss_value)
        print("Total Loss of iter", iteration, "is", total_loss)
    print("Tuning Completed!")
    vectors = sess.run(W1 + b1)

print("TSNE")
vectors = TSNE(n_components=2).fit_transform(vectors)

print("Plot")
fig = plt.figure(figsize=(14, 14))
plt.plot()
ax = fig.add_subplot(111)
for i in range(100):
    print(int2tok[i], vectors[i][0], vectors[i][1])
    plt.plot(vectors[i][0], vectors[i][1])
    ax.annotate(get_display(arabic_reshaper.reshape(int2tok[i])), xy=vectors[i], fontsize=8)

plt.savefig('word_plot.png', bbox_inches='tight')
