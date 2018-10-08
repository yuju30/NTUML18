import sys
import numpy as np
import pandas as pd
from keras.models import load_model

def readtest(filename):
	testX = pd.read_csv(filename,header=0).as_matrix()
	Xtest=[]
	for i in testX:
		Xtest.append(i[1].split())
	Xtest = np.array(Xtest).astype('float')
	return Xtest

test_file = sys.argv[1]	
Xtest = readtest(test_file)
#print(Xtest.shape) 
Xtest = Xtest.reshape((-1,48,48,1))
Xtest = Xtest/255

model = load_model(sys.argv[2])
res = model.predict(Xtest)
res = np.argmax(res, axis = 1)
ans_file = open(sys.argv[3],"w")
ans_file.write("id,label\n")
for i in range(len(res)):
	ans_str = str(i) + ',' + str(res[i]) + '\n'
	ans_file.write(ans_str)
ans_file.close()
