import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)
'''
#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def readtrain(filename):
	trainX = pd.read_csv(filename,header=0).as_matrix()
	Ytrain=[]
	Xtrain=[]
	for i in trainX:
		Ytrain.append(i[0])
		Xtrain.append(i[1].split())
	Ytrain = np.array(Ytrain).astype('int')
	Xtrain = np.array(Xtrain).astype('float')
	return Xtrain,Ytrain
def split(Xtrain,Ytrain,num):
	return Xtrain[:-num],Ytrain[:-num],Xtrain[-num:],Ytrain[-num:]

def random(Xtrain,Ytrain):
    r_list = np.array(range(0,len(Xtrain)))
    np.random.shuffle(r_list)
    Xtrain = Xtrain[r_list]
    Ytrain = Ytrain[r_list]
    return Xtrain,Ytrain

train_file = sys.argv[1]
mod_name = sys.argv[2]
(Xtrain,Ytrain) = readtrain(train_file)
Xtrain = Xtrain.reshape((-1, 48, 48, 1))
Xtrain = Xtrain/255.0
Ytrain = to_categorical(Ytrain, 7)
(Xtrain,Ytrain)=random(Xtrain,Ytrain)
(X_tr,Y_tr,X_va,Y_va) = split(Xtrain,Ytrain,5000)
#(X_tr, Y_tr) = shuffle(X_tr, Y_tr)

X_tr = np.concatenate((X_tr, X_tr[:, :, ::-1,:]), axis=0) # inverse
Y_tr = np.concatenate((Y_tr, Y_tr), axis=0)

deal_data = ImageDataGenerator(rotation_range=25, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=[0.75,1.25],
        horizontal_flip=True, fill_mode="nearest")
deal_data.fit(X_tr)

model = Sequential()

model.add(Convolution2D(70,(5,5),input_shape=(48,48,1)))#44*44
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))#22*22
model.add(Dropout(0.2))

model.add(Convolution2D(140,(3,3)))#20*20
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))#10*10
model.add(Dropout(0.3))

model.add(Convolution2D(280,(3,3)))#8*8
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))#4*4
model.add(Dropout(0.35))

model.add(Convolution2D(560,(3,3)))#2*2
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))#1*1
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units = 560))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(units = 480))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(units = 60))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(units = 7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])



Model_Check_Point = []
Model_Check_Point.append(ModelCheckpoint('model-{epoch:05d}-{val_acc:.5f}-{val_loss:.5f}.hdf5', monitor='val_acc', save_best_only=True,mode='auto', period=1))


model.fit_generator(deal_data.flow(X_tr, Y_tr, batch_size=128),
        validation_data=(X_va, Y_va), steps_per_epoch=len(X_tr) / 32,
        epochs=500,callbacks = Model_Check_Point)

model.save(mod_name)


#print(Ytrain.shape)
#print(Xtrain.shape)
