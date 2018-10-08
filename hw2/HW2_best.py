import sys
import numpy as np
import pandas as pd

def read_data(trainX_name,trainY_name,testX_name):
	trainX = pd.read_csv(trainX_name,header=0).as_matrix()
	#print(trainX.shape)
	trainX = trainX.astype(float)
	trainY = pd.read_csv(trainY_name,header=0).as_matrix()
	#print(trainY.shape)
	trainY = trainY.astype(int)
	testX = pd.read_csv(testX_name,header=0).as_matrix()
	#print(testX.shape)
	testX = testX.astype(float)
	

	return trainX,trainY,testX


def normalize(X_all, X_test):
    mu = (sum(X_all) / X_all.shape[0])
    sigma = np.std(X_all, axis=0)
    mu_a = np.tile(mu, (X_all.shape[0], 1))
    sigma_a = np.tile(sigma, (X_all.shape[0], 1))
    mu_t = np.tile(mu, (X_test.shape[0], 1))
    sigma_t = np.tile(sigma, (X_test.shape[0], 1))

    X_all = (X_all - mu_a) / sigma_a
    X_test = (X_test - mu_t) / sigma_t
    return X_all, X_test

def random(Xtrain,Ytrain):
	r_list = np.array(range(0,len(Xtrain)))
	np.random.shuffle(r_list)
	Xtrain = Xtrain[r_list]
	Ytrain = Ytrain[r_list]
	return Xtrain,Ytrain

def split(Xtrain,Ytrain,persent):
	t_size = int((len(Xtrain)*persent))
	return Xtrain[:t_size],Ytrain[:t_size],Xtrain[t_size:],Ytrain[t_size:]


def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return res

def train(Xtrain,Ytrain):
	lr = 0.05
	(data_len,fea_num)=Xtrain.shape
	w  = np.array([0.1]*fea_num)
	b = 0.1
	sgra = np.array([0.0]*fea_num)
	sgra_b = 0.0

	g=3000
	while (g>=0):
		z = Xtrain.dot(w) + b
		prob = sigmoid(z)

		err = Ytrain.flatten() - prob
		
		dLw = - np.dot(Xtrain.T,err)/data_len
		dLb = - (err.sum())/data_len

		if(g%100==0):
	  		loss = -np.mean(Ytrain.flatten().dot(np.log(prob))+(1 - Ytrain).flatten().dot(np.log(1-prob)))
	  		print("iteration num : ",3000-g," loss = ",loss)
	  		dea_pro = np.around(prob)
	  		acc_out = np.mean(1-np.abs(Ytrain.flatten()-dea_pro))
	  		print("iteration num : ",3000-g," accuracy = ",acc_out)
		#adagrad
		sgra = sgra + dLw**2
		sgra_b = sgra_b + dLb**2

		w = w - (lr*dLw)/np.sqrt(sgra)
		b = b - (lr*dLb)/np.sqrt(sgra_b)
		g=g-1

	return w ,b
def select(data):
	work = np.array([0, 1, 3, 4, 5])
	data = np.concatenate((data, data[:, work] ** 1,data[:, work] ** 2,data[:, work] ** 2.5,data[:, work] ** 3), axis = 1)
	
	return data

def test(test_data,w,b,ans_filename):
	ans_file = open(ans_filename, "w")
	ans_file.write("id,label\n")
	z = test_data.dot(w)+b
	prob = sigmoid(z)
	res_prob = np.around(prob)
	#xgb_Y = pd.read_csv('ans_xgb').as_matrix()
	#xgb_Y = xgb_Y.astype(int)
	#xgb_Y = xgb_Y[:,1]
	#acc_res = np.mean(1-np.abs(xgb_Y.flatten()-res_prob))
	for i in range(len(res_prob)):
		ans_str = str(i+1) + ',' + str(int(res_prob[i])) + '\n'
		ans_file.write(ans_str)
	#print("Final result : ",acc_res)

def judge(test_data,X_va,Y_va,w,b):
	z = test_data.dot(w)+b
	prob = sigmoid(z)
	res_prob = np.around(prob)
	xgb_Y = pd.read_csv('ans_xgb').as_matrix()
	xgb_Y = xgb_Y.astype(int)
	xgb_Y = xgb_Y[:,1]
	xgb_Y = xgb_Y.reshape(-1,1)
	print(xgb_Y.shape,Y_va.shape)
	acc_res = np.mean(1-np.abs(xgb_Y.flatten()-res_prob))
	print("temp result : ",acc_res)
	return acc_res
		
trainX_name = sys.argv[1]
trainY_name = sys.argv[2]
testX_name = sys.argv[3]
ans_name = sys.argv[4]
max_acc = 0
max_w = []
max_b = 0

(trainX_ret,trainY_ret,testX_ret) = read_data(trainX_name,trainY_name,testX_name)

Xtrain = select(trainX_ret)
Xtest = select(testX_ret)

(Xtrain_ret,Xtest_ret) = normalize(Xtrain,Xtest)
'''
for g in range(10):
	(Xtrain_r,Ytrain_r) = random(Xtrain_ret,trainY_ret)
	(X_tr,Y_tr,X_va,Y_va) = split(Xtrain_r,Ytrain_r,0.7)
	(w_res, b_res) = train(X_tr, Y_tr)
	acc = judge(Xtest_ret,X_va,Y_va,w_res,b_res)
	if(acc > max_acc):
		max_acc = acc
		max_w = w_res
		max_b = b_res
	print("currest best acc : ",max_acc)
# save model
np.save('model.npy',(max_w,max_b))
# read model
'''
(w,b) = np.load('model_best.npy')

test(Xtest_ret,w,b,ans_name)
