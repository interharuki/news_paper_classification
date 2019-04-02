# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:28:32 2018

@author: inter
"""

import MeCab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import csv
def write_csv(filename, csv_list):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)
    f.close()
    


def _split_to_words(text):
    tagger = MeCab.Tagger('-O wakati')
    try:
        res = tagger.parse(text.strip())
    except:
        return []
    
    return res.split()

tagger = MeCab.Tagger("")
def japanese_analyzer(string):
    result_list = []
    for line in tagger.parse(string).split("\n"):
        splited_line = line.split("\t")
        if len(splited_line) >= 2 and "名詞" in splited_line[1]:
            result_list.append(splited_line[0])
    return result_list

def get_vector_by_text_list(_items):
    #count_vect = CountVectorizer(analyzer=_split_to_words)
    count_vect = TfidfVectorizer(analyzer=_split_to_words)
    #count_vect = CountVectorizer()
    bow = count_vect.fit_transform(_items)
    
    """単語の数え上げ"""
        #CountVectorizerの実行
    txt_vec = CountVectorizer(analyzer=_split_to_words)
    txt_vec.fit(_items)
     
    #抽出した単語を確認する
    txt_vec.get_feature_names()
     

    #特徴量の抽出
    word = txt_vec.transform(_items)
   
     
    #特徴量ベクトルに変換（出現頻度）
    vector = word.toarray()

    vector =  np.sum(vector,axis = 0)
     
    #単語の出現頻度を確認
    word_list = []
    for word,count in zip(txt_vec.get_feature_names()[:], vector[:]):
        #print(word, count)
        word_list.append([word,count])

    write_csv('wordlist_yomiuri.csv',word_list)
    """
    print(_items)
    item_list = []
    for i in range (len(_items)):
        item_list.append(_split_to_words(_items[i]))
    
    print(item_list)
    



    
    count_vect = TfidfVectorizer()
    #count_vect = CountVectorizer()
    bow = count_vect.fit_transform(item_list)
    """


    """上手く言ったやつ"""
    """
    count_vect = TfidfVectorizer(analyzer=_split_to_words)
    bow = count_vect.fit_transform(_items)
    """

    #print(bow)
    #print(count_vect.vocabulary_)
    

   
    X = bow.todense()
    return [X,count_vect]


if __name__ == '__main__':
    text='私は犬が好き'
    
    print(_split_to_words(text))

    

    

    #X,vector = get_vector_by_text_list(text)

    
    #print(_split_to_words(text))
    df = pd.read_csv('data_syasetu(without_mainichi).csv',names=('text','category'),encoding='shift_jis')
    print(df['category'][1360:2038])
    
    X,vector = get_vector_by_text_list(df['text'][1360:2038])
    #print(get_vector_by_text_list(df['text']))
    #print(X)
    #print(vector)