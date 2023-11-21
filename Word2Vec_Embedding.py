from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
import csv

## Open the Original Data File:
with open('./hiv.txt') as seq_handle:
    seqs = [seq.rstrip() for seq in seq_handle]

## Config:
leastshow = 1 # min count
k = 5 # length of word
size = 100 # dimension of features
threads = 20 # coworking threads

## Segmentation of the Original Data Text:
def text_processing(k, seqs_train, dimension, leastshow):
    sentence = []
    for S in seqs_train:
        seq_k_mer = []
        for i in range(len(seqs_train[1]) - k + 1):
            seq_k_mer.append(S[i:i + k])
        sentence.append(seq_k_mer)
    return sentence

## Word2Vec Training Model:
def w2v_train_model(word_segmentation):
    dna_w2v = Word2Vec(word_segmentation, vector_size=size, min_count=leastshow, window=5, workers=threads)
    dna_w2v.save('./DNA_W2V')
    dna_w2v.wv.save_word2vec_format('./dna_w2v_dict.csv', binary=False)

## Get Single Sentence Vector:
def get_sentence_vector(model_w2v, sentence):
    words = list(model_w2v.wv.key_to_index.keys())
    word_vectors = [model_w2v.wv[w] for w in sentence if w in words]
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector

## Get Single Sentence Vector:
def get_tfidf_weighted_sentence_vector(model_w2v, sentence):
    words = list(model_w2v.wv.key_to_index.keys())
    word_vectors = [model_w2v.wv[w] for w in sentence if w in words]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentence)
    word_tfidf = dict(zip(vectorizer.get_feature_names_out(), 
                          tfidf_matrix.toarray()[0]))
    weighted_vectors = [word_vectors[i] * word_tfidf[w] for i, w in 
                        enumerate(sentence) if w in words and len(w) == k]
    sentence_vector = np.mean(weighted_vectors, axis=0)
    return sentence_vector

## Generate Vectors of All DNAs:
if __name__ == '__main__':
    word_segmentation = text_processing(k, seqs, size, leastshow)
    w2v_train_model(word_segmentation)
    
    with open('./vec_dna_w2v.csv', 'w+', newline='') as outfile:  # Basic W2V Vectors
        writer = csv.writer(outfile)
        model_w2v = Word2Vec.load('./DNA_W2V')
        for each_sentence in word_segmentation:
            sentence_vec = get_sentence_vector(model_w2v, each_sentence)
            writer.writerow(sentence_vec)
    
    with open('./vec_dna_tfidf_w2v.csv', 'w+', newline='') as outfile:  # W2V Vectors with Tf-idf Weights
        writer = csv.writer(outfile)
        model_w2v = Word2Vec.load('./DNA_W2V')
        for each_sentence in word_segmentation:
            sentence_vec = get_tfidf_weighted_sentence_vector(model_w2v, each_sentence)
            writer.writerow(sentence_vec)
