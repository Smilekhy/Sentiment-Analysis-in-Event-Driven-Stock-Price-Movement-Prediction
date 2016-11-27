import numpy as np
import random
import operator
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer

def value2Categorical(y, clusters=2):
    label = np.copy(y)
    label[y<np.percentile(y, 100/clusters)] = 0
    for i in range(1, clusters):
        label[y>np.percentile(y, 100*i/clusters)] = i
    return label

def get_feature_label_tsne(dimensions=2, clusters=2):
    data = np.genfromtxt('./input/featureMatrix.csv')
    np.random.shuffle(data)
    print data

    X, y = data[:, :-1], data[:, -1]
    label = to_categorical(value2Categorical(y, clusters).astype('int'))

    # embedding using tf-idf and tsne
    # Adopted from https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/tfidf_tsne.py
    transformer = TfidfTransformer()
    A = transformer.fit_transform(X)
    A = A.toarray()

    tsne = TSNE(dimensions)
    X = tsne.fit_transform(A)

    validation_ratio = 0.3
    # create train and test sets
    D = int(data.shape[0] * validation_ratio)  # total number of test data

    X_train, y_train, X_test, y_test = X[:-D], label[:-D,:], X[-D:], label[-D:,:]
    return X_train, y_train, X_test, y_test


def trainTSNE(X_train, y_train, X_test, y_test, clusters=2, embedLayer=200, middle = 100):
    top_words = 2001
    lossType = 'binary_crossentropy' if y_test.shape[1] == 2 else 'categorical_crossentropy'
    model = Sequential()
    # model.add(Embedding(top_words, embedLayer, input_length=X_train.shape[1]))
    model.add(Convolution1D(nb_filter=embedLayer, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(middle, activation='relu'))
    model.add(Dense(clusters, activation='sigmoid'))
    model.compile(loss=lossType, optimizer='adam', metrics=['accuracy'])
    # return model

    print(model.summary())

    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


# def tsne_on_wikipedia():
#     sentences, word2idx = get_wikipedia_data('file', 5000, by_paragraph=True)
#     with open('w2v_word2idx.json', 'w') as f:
#         json.dump(word2idx, f)

#     # build term document matrix
#     V = len(word2idx)
#     N = len(sentences)
#     print V, N

#     # create raw counts first
#     A = np.zeros((V, N))
#     j = 0
#     for sentence in sentences:
#         for i in sentence:
#             A[i,j] += 1
#         j += 1
#     print 'finished getting raw counts'

#     transformer = TfidfTransformer()
#     A = transformer.fit_transform(A)
#     A = A.toarray()

#     idx2word = {v:k for k, v in word2idx.iteritems()}

#     # plot the data in 2-D
#     tsne = TSNE()
#     Z = tsne.fit_transform(A)
#     print 'Z.shape:', Z.shape
#     # plt.scatter(Z[:,0], Z[:,1])
#     # for i in xrange(V):
#     #     try:
#     #         plt.annotate(s=idx2word[i].encode('utf8'), xy=(Z[i,0], Z[i,1]))
#     #     except:
#     #         print 'bad string:', idx2word[i]
#     # plt.show()

#     We = Z
#     # find_analogies('king', 'man', 'woman', We, word2idx)
#     find_analogies('france', 'paris', 'london', We, word2idx)
#     find_analogies('france', 'paris', 'rome', We, word2idx)
#     find_analogies('paris', 'france', 'italy', We, word2idx)    

def tsne_on_news():
    X_train, y_train, X_test, y_test = get_feature_label_tsne(dimensions=10, clusters=2)
    trainTSNE(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    tsne_on_news()