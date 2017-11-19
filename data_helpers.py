import numpy as np
import re
import itertools
from collections import Counter
import jieba
import os


def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')

# TODO \xa0也就是&nbsp; 去除
# \u3000  全角空格符。
# 修改为短句 ， 不用完整的句子去分析
def cut_sentence(words):
    start = 0
    i = 0
    sents = []
    punt_list = '，,!?。！？;；'
    # 去除\xa0
    words = " ".join(words.split())
    for word in words:
        if word in punt_list and token not in punt_list: #检查标点符号下一个字符是否还是标点
            sentence = words[start:i+1]
            sents.append(sentence)
            start = i+1
            i += 1
        else:
            i += 1
            token = list(words[start:i+2]).pop() # 取下一个字符
    if start < len(words):
        sents.append(words[start:])
    return sents

# 分词
def segmentation(content):
    word_list = []
    #  去除无用符号， 其实也可以不用去除，让其自行分析
    useless_symbol = [' ',' ',',','，','。',':','：',';','；']
    # 一句话最长 ， 大约200个词 ， 平均分词32
    max_length = 0 
    # 一段话拥有句子最多， 大约有600 ， 平均分句200
    max_sentence = 0
    for section_list in content:
        section_word_list = []
        for sentence in section_list:
            seg_list = jieba.lcut(sentence, cut_all=False)
            # 清理掉一些特殊符号和空格。
            seg_list = [i for i in seg_list if i not in useless_symbol] 
            # print(seg_list)
            section_word_list.append(seg_list)
            if(len(seg_list) > max_length):
                max_length = len(seg_list)
                # print(seg_list)
                # print(max_length)
        word_list.append(section_word_list)
        if(len(section_list) > max_sentence):
            max_sentence = len(section_list)
            # print(max_sentence)
            # print(section_list)
    return word_list,max_sentence,max_length

# 加载本地的词汇表，如果不存在，则创建本地的词汇表。
def load_vocab(vocab_file_path, word_list,vacabulary_table_length=10000):
    if not os.path.exists(vocab_file_path):
        # 构建词汇表
        all_words = []
        for section in word_list:
            for sentence in section:
                all_words.extend(sentence)
        couter = Counter(all_words)
        couter_pairs = couter.most_common(vacabulary_table_length - 1)
        most_words , _ = list(zip(*couter_pairs))
        most_words = ['<PAD>'] + list(most_words)
        open_file(vocab_file_path, mode='w').write('\n'.join(most_words) + '\n')
        return most_words
    else:
        # 读取词汇表
        most_words = open_file(vocab_file_path).read().strip().split('\n')
        return most_words

# 将句子转换为 词汇ID表示的向量
def change_words_to_array(word_list,most_words,sentence_max_size,segmentation_max_size):
    word_dict = {}
    for i in range(len(most_words)):
        word_dict[most_words[i]] = i
    word_array = []
    for section in word_list:
        section_array = []
        sentence_count = len(section)
        for j in range(sentence_max_size if sentence_count>sentence_max_size else sentence_count):
            sentence = section[j]
            sentence_arry = []
            segmentation_count = len(sentence)
            for i in range(segmentation_max_size if segmentation_count > segmentation_max_size else segmentation_count):
                word = sentence[i]
                if word in word_dict:
                    sentence_arry.append(word_dict[word])
                else:
                    # sentence_arry.append(vacabulary_table_length)
                    sentence_arry.append(0)
            sentence_arry.extend([0 for _ in range(segmentation_count,segmentation_max_size)])
            section_array.append(sentence_arry)
        for j in range(sentence_count,sentence_max_size):
            section_array.append([0 for _ in range(segmentation_max_size)])
        word_array.append(section_array)
    return word_array

# 将label转换为向量。
def change_label_to_array(label_list):
    categories = ['体育', '财经', '房产', '家居',
        '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    category_dict = {}
    for i in range(len(categories)):
        category_dict[categories[i]] = i
    id_list = [category_dict[x] for x in label_list]
    label_array = []
    for x in id_list:
        label_array.append([1 if x == i else 0 for i in range(len(categories))])
    return label_array



# 加载数据。
def load_data_and_labels(data_file_path,setting_sentence_size,setting_segmentation_cout,vacabulary_table_length=10000):
    contents, labels = [], []
    with open_file(data_file_path) as file:
        for line in file:
            try:
                label,content = line.strip().split('\t')
                sentences = cut_sentence(content)
                labels.append(label)
                contents.append(sentences)
            except:
                pass
    word_list,sentence_max_size,segmentation_max_size = segmentation(contents)
    most_words = load_vocab("./myvocab.txt",word_list,vacabulary_table_length)
    word_array = change_words_to_array(word_list,most_words,setting_sentence_size,setting_segmentation_cout)
    label_array = change_label_to_array(labels)
    return word_array,label_array


if __name__ == '__main__':
    load_data_and_labels("./cnews.val.txt",600,100,vacabulary_table_length)