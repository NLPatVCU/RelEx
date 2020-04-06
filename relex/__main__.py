from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN
from RelEx_NN.cnn import Sentence_CNN
import numpy
import argparse
import tensorflow_hub as hu
import tensorflow as tf

# embedding_path = "home/neha/Documents/mimic3_d200.txt" model = Model(True, False) embedding=Embeddings(
# embedding_path, model) seg_cnn = Segment_CNN(model, embedding) seg_cnn.cross_validate(preceding,
# middle, succeeding, concept1, concept2, train_label)
from torch import hub



parser = argparse.ArgumentParser(prog="RelEx")
#  parser.add_argument("-d", "--data",help="provide directory location for data")
parser.add_argument("--sentrain", help="provide path to sentence training data",required=True)
parser.add_argument("--labeltrain", help="provide path to label training data",required=True)
parser.add_argument("--sentest", help="provide path to sentence testing data (if applicable)")
parser.add_argument("--labeltest", help="provide path to label testing data (if applicable)")
parser.add_argument("--embedding", help="provide path to embedding file", required=True)  # might want to make this more specific
parser.add_argument("--segment", help="activate segment-cnn", default="TRUE")
parser.add_argument("--test", help="validate model on test dataset", default="FALSE")
parser.add_argument("--multilabel", help="insert help text here", default="TRUE")
parser.add_argument("--onehot", help="create one-hot vectors", default="FALSE")
parser.add_argument("--commonwords", help="number of words to consider as features", default="10000")
parser.add_argument("--maxlen", help="maximum length of vector", default="100")
parser.add_argument("--elmo", help="use elmo embeddings", default="FALSE")
args = parser.parse_args()
segment = args.segment == "TRUE"
test = args.test == "TRUE"
multilabel = args.multilabel == "TRUE"
onehot = args.onehot == "TRUE"
elmo = args.elmo == "TRUE"
model=None

model = Model(sentrain=args.sentrain, sentest=args.sentest, labeltest=args.labeltest, labeltrain=args.labeltrain,
                  segment=segment, test=test, multilabel=multilabel, one_hot=onehot, common_words=args.commonwords,
                  maxlen=args.maxlen)

train_preceding = open("/home/neha/Desktop/P_Tr/preceding_seg").read()
train_middle = open("/home/neha/Desktop/P_Tr/middle_seg").read()
train_succeeding = open("/home/neha/Desktop/P_Tr/succeeding_seg").read()
train_concept1 = open("/home/neha/Desktop/P_Tr/concept1_seg").read()
train_concept2 = open("/home/neha/Desktop/P_Tr/concept2_seg").read()
train_sent = open("/home/neha/Desktop/P_Tr/sentence_train").read()
sents = train_sent.split("\n")
print(len(sents))


#print(args.sentrain)




embedding_path = args.embedding
#
embedding=Embeddings(embedding_path, model)
if elmo:
    seg_cnn = Sentence_CNN(model, elmo=args.sentrain, cross_validation=True, embedding=embedding)
else:
    seg_cnn = Sentence_CNN(model, cross_validation=True, embedding=embedding)



