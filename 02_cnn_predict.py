# coding: utf-8

from __future__ import print_function
import pandas as pd
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from cnn_utils import read_category, read_vocab, open_file


base_dir = 'data/new_data'
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'model/checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:

    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = message
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':

    # 恢复模型
    cnn_model = CnnModel()
    print("正在读取测试数据..")
    test_file = 'data/new_data/test_set.csv'
    with open_file(test_file) as f:
        test_data = pd.read_csv(f)
    # articles = test_data['word_seg'].apply(lambda x: x.split(' ')).tolist()
    articles = test_data['word_seg'].apply(lambda x: x.split(' ')).tolist()

    # 进行预测
    print("正在预测...")
    class_list = []
    for i in articles:
        # print(cnn_model.predict(i))
        class_list.append(cnn_model.predict(i))

    test_data['class'] = class_list
    sub_data = test_data[['id', 'class']]
    sub_data.to_csv('data/new_data/02_sub_cnn.csv', index=False)
    print("结果以保存：data/new_data/02_sub_cnn.csv")
