import numpy as np
import pandas as pd
from keras.models import Model,Sequential
from keras.layers import Flatten,add,Dot,Input,Dense,Lambda,Reshape,Dropout,Embedding
import sys
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.initializers import Zeros
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
import keras.backend as K


def rmse(y_true, y_pred):
    # y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))*rat_std

def load_data(trainfile, testfile):
	train_file = pd.read_csv(trainfile)
	test_file = pd.read_csv(testfile)

	tr_usr = train_file['UserID']
	te_usr = test_file['UserID']

	tr_mov = train_file['MovieID']
	te_mov = test_file['MovieID']

	Usr = pd.concat([tr_usr, te_usr])
	Mov = pd.concat([tr_mov,te_mov])
	Usr = Usr.unique()
	Mov = Mov.unique()
	num_users=len(Usr)
	num_movies=len(Mov)

	Dic_Usr = {}
	Dic_Mov = {}

	for i in range(num_users):
		Dic_Usr[str(Usr[i])]=i
	for j in range(num_movies):
		Dic_Mov[str(Mov[j])]=j
	np.save('Dic_Usr', Dic_Usr)
	np.save('Dic_Mov', Dic_Mov)

	tr_usr = tr_usr.apply(lambda x: Dic_Usr[str(x)])
	tr_mov= tr_mov.apply(lambda x: Dic_Mov[str(x)])
		
	return tr_usr.values,tr_mov.values,train_file['Rating'].values,num_users,num_movies

def model_construct(n_users,n_items,lat_dim):
	
	usr_input = Input(shape=[1])
	item_input = Input(shape=[1])
	usr_vec = Embedding(n_users,lat_dim,embeddings_regularizer=l2(0.00001))(usr_input)
	#usr_vec = Flatten()(usr_vec)
	usr_vec = Reshape((lat_dim,))(usr_vec)
	usr_vec = Dropout(0.1)(usr_vec)
	item_vec = Embedding(n_items,lat_dim,embeddings_regularizer=l2(0.00001))(item_input)
	#item_vec = Flatten()(item_vec)
	item_vec = Reshape((lat_dim,))(item_vec)
	item_vec = Dropout(0.1)(item_vec)
	
	usr_bias = Embedding(n_users,1,embeddings_regularizer=l2(0.00001))(usr_input)
	usr_bias = Flatten()(usr_bias)
	item_bias = Embedding(n_items,1,embeddings_regularizer=l2(0.00001))(item_input)
	item_bias = Flatten()(item_bias)

	r_h = Dot(axes=-1)([usr_vec,item_vec])
	r_h = add([r_h,usr_bias,item_bias])
	model = Model(inputs=[usr_input,item_input],outputs=[r_h])
	
	model.summary()
	model.compile(loss='mse',optimizer='adam')

	return model


tr_file = sys.argv[1]
te_file = sys.argv[2]
model_name = sys.argv[3]

train_Usr,train_Mov,train_rat,num_users,num_movies=load_data(tr_file,te_file)

print('num_users : ',num_users)
print('num_movies : ',num_movies)


index = np.random.permutation(len(train_Usr))
train_Usr,train_Mov, train_rat = train_Usr[index],train_Mov[index],train_rat[index]


rat_mean = np.mean(train_rat, axis = 0)
rat_std= np.std(train_rat, axis = 0)
np.save("normal.npy", [rat_mean, rat_std])
train_rat = (train_rat - rat_mean) / rat_std
print('train_rat : ',len(train_rat))

callbacks = []
callbacks.append(EarlyStopping(monitor='val_rmse', patience=10))
callbacks.append(ModelCheckpoint(model_name, monitor='val_rmse', save_best_only=True))

model = model_construct(num_users, num_movies,15)
model.compile(loss='mse', optimizer='adam', metrics=[rmse])
model.fit([train_Usr, train_Mov], train_rat, epochs=200, batch_size=1024, validation_split=0.1, callbacks=callbacks) 



