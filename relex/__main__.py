from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN
import numpy
import argparse

# embedding_path = "home/neha/Documents/mimic3_d200.txt" model = Model(True, False) embedding=Embeddings(
# embedding_path, model) seg_cnn = Segment_CNN(model, embedding) seg_cnn.cross_validate(model.preceding,
# model.middle, model.succeeding, model.concept1, model.concept2, model.train_label)

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

args = parser.parse_args()
segment = args.segment == "TRUE"
test = args.test == "TRUE"
multilabel = args.multilabel == "TRUE"
onehot = args.onehot == "TRUE"
model=None

model = Model(sentrain=args.sentrain, sentest=args.sentest, labeltest=args.labeltest, labeltrain=args.labeltrain,
                  segment=segment, test=test, multilabel=multilabel, one_hot=onehot, common_words=args.commonwords,
                  maxlen=args.maxlen)


embedding_path = args.embedding
# model = Model(True, False)
embedding=Embeddings(embedding_path, model)
seg_cnn = Segment_CNN(model, embedding, cross_validation=True)

