#!/usr/bin/python
import os
import json
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import en
import operator
from datetime import datetime
from sklearn.utils import shuffle
from nltk.corpus import reuters


# reference https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/glove.py

class Glove:
    def __init__(self, D, V, context_sz):
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=10e-5, reg=0.1, xmax=100, alpha=0.75, epochs=10, gd=False, use_theano=True):
        # build co-occurrence matrix
        # paper calls it X, so we will call it X, instead of calling
        # the training data X
        # TODO: would it be better to use a sparse matrix?
        t0 = datetime.now()
        V = self.V
        D = self.D

        # co occurence matrix 로드 혹은 생성 후 저장
        if os.path.exists(cc_matrix):
            X = np.load(cc_matrix)
        else:
            # Co occurrence 관리할 matrix X (V,V)
            X = np.zeros((V, V))
            N = len(sentences)
            print("number of sentences to process:", N)
            it = 0
            for sentence in sentences:
                it += 1
                # 상태 표시
                if it % 10000 == 0:
                    print( "processed", it, "/", N)
                n = len(sentence)
                for i in range(n):
                    wi = sentence[i]

                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_sz)

                    # we can either choose only one side as context, or both
                    # here we are doing both

                    # make sure "start" and "end" tokens are part of some context
                    # otherwise their f(X) will be 0 (denominator in bias update)

                    # context_size가 채워지지 않은 문장 초반 구간
                    # 1/(i+1) 의 점수를 co occurrence matrix에 추가함
                    if i - self.context_sz < 0:
                        points = 1.0 / (i + 1)
                        X[wi,0] += points
                        X[0,wi] += points
                    # 양쪽 컨텍스트를 고려할 때,
                    if i + self.context_sz > n:
                        points = 1.0 / (n - i)
                        X[wi,1] += points
                        X[1,wi] += points

                    # 앞쪽 context만 고려하는 방식 선택
                    for j in range(start, i):

                        # i,j 같으면 스킵
                        if j == i: continue

                        # wj 에 단어 번호 할당
                        wj = sentence[j]

                        # 1 / (i,j거리) 를 포인트로 계산
                        points = 1.0 / abs(i - j) # this is +ve

                        # co occurrence matrix 의 해당 부분에 point 합산
                        X[wi,wj] += points
                        X[wj,wi] += points

            # save the cc matrix because it takes forever to create
            np.save(cc_matrix, X)

        print("max in X:", X.max())

        # weighting
        fX = np.zeros((V, V))
        # Co occurrence matrix normalized
        # xmax = 100, 빈도수 100보다 높은 것 1으로 clipping, 미만인 것들은 xmax로 나눠줌
        fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha
        fX[X >= xmax] = 1

        print("max in f(X):", fX.max())

        # target
        logX = np.log(X + 1)

        print("max in log(X):", logX.max())

        print("time to build co-occurrence matrix:", (datetime.now() - t0))

        # logX, fX 를 input으로 넣어서 GD로 학습된 W, U matrix를 word embedding 으로 사용
        # initialize weights
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()

        if gd and use_theano:
            thW = theano.shared(W)
            thb = theano.shared(b)
            thU = theano.shared(U)
            thc = theano.shared(c)
            thLogX = T.matrix('logX')  # T means Theano
            thfX = T.matrix('fX')

            params = [thW, thb, thU, thc]

            # 학습 파트 (GloVe 수식 , Theano )
            thDelta = thW.dot(thU.T) + T.reshape(thb, (V, 1)) + T.reshape(thc, (1, V)) + mu - thLogX
            thCost = ( thfX * thDelta * thDelta ).sum()

            grads = T.grad(thCost, params)

            updates = [(p, p - learning_rate*g) for p, g in zip(params, grads)]

            train_op = theano.function(
                inputs=[thfX, thLogX],
                updates=updates,
            )

        costs = []
        sentence_indexes = range(len(sentences))
        for epoch in range(epochs):
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
            cost = ( fX * delta * delta ).sum()
            costs.append(cost)
            print("epoch:", epoch, "cost:", cost)

            if gd:
                # gradient descent method

                if use_theano:
                    train_op(fX, logX)
                    W = thW.get_value()
                    b = thb.get_value()
                    U = thU.get_value()
                    c = thc.get_value()

                else:
                    # update W
                    oldW = W.copy()
                    for i in range(V):
                        W[i] -= learning_rate*(fX[i,:]*delta[i,:]).dot(U)
                    W -= learning_rate*reg*W

                    # update b
                    for i in range(V):
                        b[i] -= learning_rate*fX[i,:].dot(delta[i,:])
                    b -= learning_rate*reg*b

                    # update U
                    for j in range(V):
                        U[j] -= learning_rate*(fX[:,j]*delta[:,j]).dot(oldW)
                    U -= learning_rate*reg*U

                    # update c
                    for j in range(V):
                        c[j] -= learning_rate*fX[:,j].dot(delta[:,j])
                    c -= learning_rate*reg*c

            else:
                # ALS method

                # update W
                # fast way
                # t0 = datetime.now()
                for i in range(V):
                    # matrix = reg*np.eye(D) + np.sum((fX[i,j]*np.outer(U[j], U[j]) for j in xrange(V)), axis=0)
                    matrix = reg*np.eye(D) + (fX[i,:]*U.T).dot(U)
                    # assert(np.abs(matrix - matrix2).sum() < 10e-5)
                    vector = (fX[i,:]*(logX[i,:] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(matrix, vector)
                # print "fast way took:", (datetime.now() - t0)

                # update b
                for i in range(V):
                    denominator = fX[i,:].sum()
                    # assert(denominator > 0)
                    numerator = fX[i,:].dot(logX[i,:] - W[i].dot(U.T) - c - mu)
                    # for j in xrange(V):
                    #     numerator += fX[i,j]*(logX[i,j] - W[i].dot(U[j]) - c[j])
                    b[i] = numerator / denominator / (1 + reg)
                # print "updated b"

                # update U
                for j in range(V):
                    matrix = reg*np.eye(D) + (fX[:,j]*W.T).dot(W)
                    vector = (fX[:,j]*(logX[:,j] - b - c[j] - mu)).dot(W)
                    U[j] = np.linalg.solve(matrix, vector)

                # update c
                for j in range(V):
                    denominator = fX[:,j].sum()
                    numerator = fX[:,j].dot(logX[:,j] - W.dot(U[j]) - b  - mu)
                    c[j] = numerator / denominator / (1 + reg)

        self.W = W
        self.U = U

        plt.plot(costs)
        plt.show()

    def save(self, fn):
        # function word_analogies expects a (V,D) matrx and a (D,V) matrix
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)

def unify_word(word): # went -> go, apples -> apple, BIG -> big
    try: word = en.verb.present(word) # unify tense
    except: pass
    try: word = en.noun.singular(word) # unify noun
    except: pass
    return word.lower()

def get_reuters_data(n_vocab):
    # return variables
    sentences = []
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_count = {0: float('inf'), 1: float('inf')}
    tag = 0
    # field 에 로이터의 각 뉴스기사 key 할당 (총 10788개의 뉴스기사)
    for field in reuters.fileids():

        # sentence 에 각 뉴스기사가 리스트 형태로 할당됨
        # ['ASIAN', 'EXPORTERS', 'FEAR', 'DAMAGE', 'FROM', 'U', ...], length = 899, key = test/14826
        sentence = reuters.words(field)

        # 소문자, 복수->단수, 시제 현재시점으로 통일하는 등 preprocessing 작업
        # ['asian', 'exporter', 'fear', 'damage', 'from', 'u', ...]
        tokens = [unify_word(t) for t in sentence]

        for t in tokens:
            # 단어가 word2idx 에 할당되지 않은 경우
            if t not in word2idx:
                # (전체 단어 중에서)새로 들어온 단어를 word2idx 딕셔너리에 할당하고 current_idx 번호를 value 값으로 부여
                word2idx[t] = current_idx
                # current_idx 번째 list element에 새로운 단어 string 할당
                idx2word.append(t)
                # 인덱스 관리
                current_idx += 1
            # 기존에 들어간 단어의 경우 단어의 index를 가져오고
            idx = word2idx[t]
            # word_idx_count 딕셔너리에는 단어와 매치된 번호 idx를 키로가지고 value는 모든 문서에서 단어가 등장한 빈도수를 관리한다
            word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
        # 위 루프에서 처리한 하나의 문서를 대응되는 index의 형태로 변환
        sentence_by_idx = [word2idx[t] for t in tokens]
        # 전체 문서를 관리
        sentences.append(sentence_by_idx)
        tag += 1
        print(tag)

    # restrict vocab size
    # 빈도순이 높은순으로 단어 사전 정렬
    # n_vocab 번까지 높은 단어만 살리고 그 이외 단어는 날림
    # 실제 사용할 데이터역시 날린단어를 제외 하고 사용
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print(word, count)
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx 
    unknown = new_idx

    # map old idx to new idx
    sentences_small = []
    for sentence in sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    # 문장을 단어번호로 표현한 전체 문서의 small 버전, word 와 번호를 매치한 small 버전 반환
    return sentences_small, word2idx_small

def main(we_file, w2i_file, sen):
    # co occurrence matrix 저장할 파일 설정
    cc_matrix = "./input/cc_matrix.npy"
    if not os.path.isfile(w2i_file):
        sentences, word2idx = get_reuters_data(n_vocab=2000)
        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)
        with open(sen, 'w') as f:
            json.dump(sentences, f)
    else:
        with open(w2i_file) as data_file:    
            word2idx = json.load(data_file)
        with open(sen) as data_file:    
            sentences = json.load(data_file)

    # 전체 단어 개수
    V = len(word2idx)
    # Embedding dimension, total word number, context_size (앞,뒤 몇개 단어까지 고려할 것인지)
    model = Glove(50, V, 10)
    # model.fit(sentences, cc_matrix=cc_matrix, epochs=20) # ALS
    model.fit(
        sentences,
        cc_matrix=cc_matrix,
        learning_rate=3*10e-5,
        reg=0.01,
        epochs=2000,
        gd=True,
        use_theano=True
    ) # gradient descent
    model.save(we_file)


if __name__ == '__main__':
    we = './input/glove_model_50.npz'
    w2i = './input/word2idx.json'
    sen = './input/sentences.json'
    main(we, w2i, sen)
