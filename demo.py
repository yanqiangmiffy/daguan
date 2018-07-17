import pandas as pd
import time
from collections import Counter
import seaborn
from keras.preprocessing.text import Tokenizer
from util import load_data,read_train,read_test
if __name__ == '__main__':
    train_data_file = 'data/new_data/train_set.csv'
    test_data_file = 'data/new_data/test_set.csv'

    X_train, _=read_train(train_data_file)
    _, X_test=read_test(train_data_file)

    # vocab=Counter()
    # for text in X_train:
    #     for word in text.split(' '):
    #         vocab[word]+=1
    #
    # for text in X_test:
    #     for word in text.split(' '):
    #         vocab[word]+=1
    #
    # print(vocab)

    texts=X_train+X_test
    tokenizer=Tokenizer(num_words=None)
    tokenizer.fit_on_texts(texts)
    print(tokenizer.word_index)