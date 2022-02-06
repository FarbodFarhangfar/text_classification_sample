import os
import re
from math import log


def get_words(dic):  # get the words out of dictionary
    for keys, sentence in dic.items():
        word_list = sentence.split()
        temp_list = []
        for words in word_list:
            words = re.sub("[^\w]", "", words)
            temp_list.append(words)
        dic[keys] = temp_list
    return dic


def get_doc(path):  # open test files in a dir
    dic = {}
    for files in os.listdir(path):
        if files.endswith(".txt"):
            try:
                opened_file = open(path + "/" + files)
                dic[files] = opened_file.read()
            except UnicodeDecodeError:
                opened_file = open(path + "/" + files, encoding="UTF-8")
                dic[files] = opened_file.read()
    return dic


def IDF(dic):  # calculate IDF
    idf_dic = {}
    words_dic = get_words(dic)
    for keys, word_list in words_dic.items():
        result = []
        word_list_len = len(word_list)
        for words in word_list:
            word_result = {}
            if words not in word_result:
                word_count = word_list.count(words)
                word_idf = word_count / word_list_len
                word_result[words] = word_idf
            result.append(word_result)
        idf_dic[keys] = result
    return idf_dic


def TF(dic):  # calculate TF
    tf_dic = {}
    dic_list = len(dic)
    words_dic = get_words(dic)
    for keys, word_list in words_dic.items():
        found_word = []
        for words in word_list:
            if words not in tf_dic:
                tf_dic[words] = 1
                found_word.append(words)
            else:
                if words not in found_word:
                    tf_dic[words] += 1
    for keys, num in tf_dic.items():
        tf_dic[keys] = log(dic_list / num, 10)
    return tf_dic


def frequency(word, dic):  # get frequency
    count = 0
    for keys, word_list in dic.items():
        if word in word_list:
            count = count + word_list.count(word)
    result = count / len(dic)
    return result


def matching(query, doc):  # match words in query and document
    result = []
    for key, word_list in doc.items():
        res = set(word_list).intersection(query)
        for words in res:
            result.append(result)
    return result


doc = get_doc("E:\SAMPLES\samplestxt")
query = get_doc("E:\SAMPLES/New folder (3)")
doc_idf = doc.copy()
doc_tf = doc.copy()
matched = matching(query, doc)
idf = IDF(doc_idf)
tf = TF(doc_tf)
print("words tf :", tf)
print("-----------------------------")
print("words idf :", idf)
print("-----------------------------")
result = []
query_list = []
for key, value in query.items():
    query_list.append(value)

for words in matched:
    print(words)
    res = 0
    for keys, words_list in idf:
        for i in words_list:
            res = tf[words] * i[words] * (len(doc) + 1) / frequency(words, doc)
    result.append(res)

result.sort()

for i in range(10):
    print(result[i])
