"""
    加载数据
"""
import sys
import numpy as np
import pandas as pd
from collections import Counter
import tensorflow.contrib.keras as kr


def open_file(filename, mode='r'):
    """
    常用文件操作
    :param filename: 文件名称
    :param mode: 读取模式
    :return:
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """
    读取文件数据
    :param filename:
    :return:
    """
    with open_file(filename) as f:
        data = pd.read_csv(filename)
    articles = data['word_seg'].apply(lambda x: x.split(' ')).tolist()
    # articles = data['word_seg'].apply(lambda x: x.split(' ')).tolist()
    labels = data['class'].tolist()

    return articles, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """
    根据训练集创建词汇表
    :param train_dir:
    :param vocab_dir:
    :param vocab_size:
    :return:
    """
    print("正在构建词汇表")
    data_train, _ = read_file(train_dir)

    all_data = []
    for article in data_train:
        all_data.extend(article)

    counter = Counter(all_data)
    print("单词总个数：",len(counter))

    count_pairs = counter.most_common(vocab_size - 1)[800:]
    print(count_pairs)
    words, _ = list(zip(*count_pairs))

    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """
    读取词汇表
    :param vocab_dir:
    :return:
    """
    words = open_file(vocab_dir).read().strip().split('\n')
    # 将单词转为索引
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """
    读取分类目录，固定
    :return:
    """
    categories=[str(i+1) for i in range(19)]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """
    将id表示的article转为文字
    :param content:
    :param words:
    :return:
    """
    return ' '.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=900):
    """
    将article转换为id表示
    :param filename:
    :param word_to_id:
    :param cat_to_id:
    :param max_length:
    :return:
    """
    articles, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(articles)):
        data_id.append([word_to_id[x] for x in articles[i] if x in word_to_id])
        label_id.append(cat_to_id[str(labels[i])])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == '__main__':
    # build_vocab('data/new_data/train_set.csv','data/new_data/vocab.txt',vocab_size=6000)
    # read_vocab('data/new_data/vocab.txt')
    # read_category()
    pass
