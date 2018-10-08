import sys
import numpy as np
import pandas as pd
from keras.models import Sequential,Model
from gensim.models import Word2Vec
from keras.layers import Input,LSTM,Bidirectional,Flatten, GRU, Dropout, Dense,TimeDistributed, Activation
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers
import _pickle as pk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import regularizers
import gensim

def loaddata(file_label,file_nolab):
	label = []
	word_train = []
	lab_data = []
	la_data = open(file_label,"r",encoding='utf-8')
	no_la = open(file_nolab,"r",encoding='utf-8')
	for l in la_data:
		tmp = l.strip().split(" +++$+++ ")
		label.append(int(tmp[0]))
		word_train.append(tmp[1])
		lab_data.append(tmp[1])
	for n in no_la:
		tmp1 = n.strip()
		word_train.append(tmp1)
	label = np.array(label)
	word_train = np.array(word_train)
	lab_data = np.array(lab_data)
	#print("yes/no label :",len(label),"tr_data : ",len(word_train))
	return label,word_train,lab_data

def random(Xtrain,Ytrain):
    r_list = np.array(range(0,len(Xtrain)))
    np.random.shuffle(r_list)
    Xtrain = Xtrain[r_list]
    Ytrain = Ytrain[r_list]
    return Xtrain,Ytrain
def split_data(X,Y, ratio):
	data_size = len(X)
	val_size = int(data_size * ratio)
	return X[val_size:],Y[val_size:],X[:val_size],Y[:val_size]

file_label = sys.argv[1]
file_nolab = sys.argv[2]
tok_path = sys.argv[3]
Word_path = sys.argv[4]
model_path = sys.argv[5]
(tr_lab,word_data,tr_data) = loaddata(file_label,file_nolab)
#print("num 0 ",tr_lab[0],tr_data[0])
print("label :",tr_lab.shape,"tr_data : ",tr_data.shape)
Max_len = 40

stem = gensim.parsing.porter.PorterStemmer()
tr_data = [e for e in stem.stem_documents(tr_data)]
word_data = [k for k in stem.stem_documents(word_data)]

tokenizer = Tokenizer(num_words=None, filters='\t\n')
tokenizer.fit_on_texts(word_data)

pk.dump(tokenizer,open(tok_path,'wb'))
tokenizer = pk.load(open(tok_path,'rb'))
(tr_data_f,tr_lab_f,va_data,va_lab) = split_data(tr_data,tr_lab,0.1)

sequences = tokenizer.texts_to_sequences(tr_data_f)
data = np.array(pad_sequences(sequences, maxlen=Max_len))
val_sequences = tokenizer.texts_to_sequences(va_data)
valid_data = np.array(pad_sequences(val_sequences, maxlen=Max_len))


#labels = np.array(to_categorical(tr_lab))
#labels = tr_lab


#print("tr_data_f,tr_lab_f,va_data,va_lab : ",tr_data_f.shape,tr_lab_f.shape,va_data.shape,va_lab.shape)

word2vec_data = [w.split(" ") for w in word_data]
print("=============Word2Vec=============")
WVmodel = Word2Vec(word2vec_data, size=100, window=5, min_count=0, workers=4)
WVmodel.save(Word_path)
R_WVmodel = Word2Vec.load(Word_path)


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print("Sequence 0 :",sequences[0])
print("tr_data 0 : ",tr_data[0])
len_tr = len(word2vec_data)

#translate
embeded = np.zeros((len(word_index),100))
cou = 0
for w ,i in word_index.items():
	try:
		tmp = R_WVmodel.wv[w]
		embeded[i] = tmp
	except:
		cou+=1
#train
inputs = Input(shape=(Max_len,))

# Embedding layer
embedding_inputs = Embedding(len(word_index),100,weights=[embeded],trainable=False)(inputs)
# RNN 
RNN_cell_f = Bidirectional(LSTM(128,activation="tanh",dropout=0.3,return_sequences = True))(embedding_inputs)
RNN_cell = Bidirectional(LSTM(50,activation="tanh",dropout=0.2,return_sequences = False))(RNN_cell_f)

#RNN_cell= LSTM(128,dropout=0.3,return_sequences = False)
#RNN_output = RNN_cell(embedding_inputs)
# DNN layer
outputs = Dense(50,activation='relu',kernel_regularizer=regularizers.l2(0.1))(RNN_cell)
outputs = Dropout(0.3)(outputs)
outputs = Dense(1, activation='sigmoid')(outputs)
    
model =  Model(inputs=inputs,outputs=outputs)

model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

Model_Check_Point = []
Model_Check_Point.append(ModelCheckpoint('model-{epoch:05d}-{val_acc:.5f}-{val_loss:.5f}.hdf5', monitor='val_acc', save_best_only=True,mode='auto', period=1))
#for i in rangwe(3):
model.summary()
model.fit(data,tr_lab_f,validation_data=(valid_data,va_lab) ,batch_size=64, epochs=10,callbacks = Model_Check_Point)
model.save(model_path)
