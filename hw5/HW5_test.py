import sys
import numpy as np
import pandas as pd

from keras.models import Sequential
from gensim.models import Word2Vec
from keras.layers import Flatten, GRU, Dropout, Dense,TimeDistributed, Activation
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers
import _pickle as pk
import gensim
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def loaddata(file_test):
	data = []
	test_data = open(file_test,"r",encoding='utf-8')
	for l in test_data:
		tmp = l.strip('\n').split(",",1)
		data.append(tmp[1])
	data = np.array(data)

	return data

file_test = sys.argv[1]
model_1_path = sys.argv[3]
#model_2_path = sys.argv[4]
#Word_path = sys.argv[4]
tok_path = sys.argv[4]
te_data = loaddata(file_test)
te_data = te_data[1:]
#print("data [0] :",te_data[0],"data [1] :",te_data[1])
Max_len=40

stem = gensim.parsing.porter.PorterStemmer()
te_data = [e for e in stem.stem_documents(te_data)]


#WVmodel = Word2Vec.load(Word_path)
model_1 = load_model(model_1_path)
#model_2 = load_model(model_2_path)
tokenizer = pk.load(open(tok_path,'rb'))

sequences = tokenizer.texts_to_sequences(te_data)
data = pad_sequences(sequences, maxlen=40)



'''
len_te = len(te_data)
#translate
embeded = np.zeros((len_te,Max_len,80))
for i in range(len_te):
	for j in range(len(te_data[i])):
		if te_data[i][j] in WVmodel.wv.vocab:
			embeded[i][j] = WVmodel.wv[te_data[i][j]]
'''
ans_1 = model_1.predict(data)
#ans_2 = model_2.predict(embeded, batch_size = 256)
ans_1 = ans_1.flatten()
#ans_2 = ans_2.flatten()
#ans = (ans_1+ans_2)/2
ans = ans_1
ans = np.around(ans)

ans_file = open(sys.argv[2],"w")
ans_file.write("id,label\n")
for i in range(len(ans)):
	ans_str = str(i) + ',' + str(int(ans[i]))+ '\n'
	ans_file.write(ans_str)
ans_file.close()


