import os

import numpy
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_hub as hu
import threading
import pickle
import warnings
from sklearn.metrics.pairwise import cosine_similarity
# elmo_model = hu.Module("https://tfhub.dev/google/elmo/3", trainable=False)
# source = "/home/neha/Desktop/P_Tr/sentence_train"


# Fetch ELMo Embeddings
def Get_ELMo_Embeddings(x):
    embeddings = elmo_model(x, signature="default", as_dict=True)
    # word_emb   = embeddings["word_emb"]
    # lstm_opt1  = embeddings["lstm_outputs1"]
    # lstm_opt2  = embeddings["lstm_outputs2"]
    # default    = embeddings["default"]

    # Returns The Weighted Sum Of All 3 Layers For Each Token Per Sentence
    #   Return shape: [batch_size, max_sequence_length, 1024]
    return embeddings["elmo"]


# Convert Tensor Into A Regular/Numpy Array

def Convert_Tensor_Into_Numpy_Array(tensor):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        # This Is Where The Conversion Happens In The Tensorflow Session
        return sess.run(tensor)

def Embed_To_File(sents, start, end, currentThread, totalThread):

    if(currentThread==totalThread-1):
        end=len(sents)


    batch = sents[start:end]

    for i in range(len(batch)):
        print(tf.gfile.IsDirectory("https://tfhub.dev/google/elmo/3"))
        elmo_model = hu.Module("https://tfhub.dev/google/elmo/3", trainable=False)
        embed = elmo_model([batch[i]], signature="default", as_dict=True)["elmo"]
        embedArr = Convert_Tensor_Into_Numpy_Array(embed)

        print("SENTENCE NUMBER "+str(i)+"/"+str(len(batch))+" OF BATCH "+str(currentThread)+" IS DONE")
        pickle_out = open("embeddingfiles/embeddingfileb"+str(currentThread)+"s"+str(i),"wb")
        pickle.dump(embedArr, pickle_out)
    print("Batch "+str(currentThread)+" is done!! :D")


def threadTime(sents, threadNum):
    count = len(sents)
    batch_size = int(len(sents)/threadNum)
    threads = []

    for i in range(0,threadNum):

        t = threading.Thread(target=Embed_To_File, name='t'+str(i), args=(sents, batch_size*i, batch_size*(i+1), i, threadNum))
        threads.append(t)
    for i in range(0,threadNum):
        threads[i].start()
    for i in range(0,threadNum):
        threads[i].join()

