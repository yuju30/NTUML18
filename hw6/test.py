import sys
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import load_model


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

def load_data(test_file, Dic_Usr,Dic_Mov):
    test_data = pd.read_csv(test_file)

    test_Usr = test_data['UserID'].apply(lambda x: Dic_Usr[str(x)])
    test_Mov = test_data['MovieID'].apply(lambda x: Dic_Mov[str(x)])

    return test_Usr.values,test_Mov.values

test_path = sys.argv[1]
ans_path = sys.argv[2]
Dic_Usr_path = sys.argv[3]
Dic_Mov_path = sys.argv[4]
model_path =sys.argv[5]
nor_path = sys.argv[6]


Dic_Usr = np.load(Dic_Usr_path)[()]
Dic_Mov = np.load(Dic_Mov_path)[()]
test_Usr,test_Mov = load_data(test_path,Dic_Usr,Dic_Mov)

model = load_model(model_path, custom_objects={'rmse': rmse})
result = model.predict([test_Usr,test_Mov]).squeeze()

nor = np.load(nor_path)
mean = nor[0]
std = nor[1]
result = result * std + mean

result = result.clip(1.0, 5.0)

ans_file = open(ans_path,"w")
ans_file.write("TestDataID,Rating\n")
for i in range(len(result)):
    ans_str = str(i+1) + ',' + str(result[i])+ '\n'
    ans_file.write(ans_str)
ans_file.close()
