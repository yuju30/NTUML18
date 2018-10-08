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
    # Feature normalization with train and test X
    #X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_all) / X_all.shape[0])
    sigma = np.std(X_all, axis=0)
    mu_a = np.tile(mu, (X_all.shape[0], 1))
    sigma_a = np.tile(sigma, (X_all.shape[0], 1))
    mu_t = np.tile(mu, (X_test.shape[0], 1))
    sigma_t = np.tile(sigma, (X_test.shape[0], 1))

    X_all = (X_all - mu_a) / sigma_a
    X_test = (X_test - mu_t) / sigma_t
    # Split to train, test again
    #X_all = X_train_test_normed[0:X_all.shape[0]]
    return X_all, X_test

def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return res

def train(Xtrain,Ytrain):
	(data_len,fea_num)=Xtrain.shape
	bel_1 = Xtrain[(Ytrain==1).flatten()]
	bel_0 = Xtrain[(Ytrain==0).flatten()]
	#print("bel_1 : ",len(bel_1)," bel_0 : ",len(bel_0))
	# [num of belong to 0 or 1] * 123 features
	mn_1 = np.mean(bel_1,axis=0)
	mn_0 = np.mean(bel_0,axis=0)
	#print("mn_1 : ",mn_1.shape," mn_0 : ",mn_0.shape)
	num1 = len(bel_1) 
	num0 = len(bel_0)
	#print("num_1 : ",num1," num_0 : ",num0)

	std_1 = np.zeros((fea_num,fea_num))
	std_0 = np.zeros((fea_num,fea_num))

	for i in range(data_len):
		if(Ytrain[i]==1):
			std_1 += np.dot(np.transpose([Xtrain[i]-mn_1]),[(Xtrain[i]-mn_1)])
			
		else:
			std_0 += np.dot(np.transpose([Xtrain[i]-mn_0]),[(Xtrain[i]-mn_0)])
			
	std_1 = std_1/num1	
	std_0 = std_0/num0
	
	std = float(num1)/data_len * std_1 + float(num0)/data_len * std_0
	#print("std_det  : ",np.linalg.det(std))
	#print("std : ",std.shape)
	return num1,num0,mn_1,mn_0,std


def select(data):
	sele = np.array([0, 1, 3, 4, 5])
	data = np.concatenate((data, data[:, sele] ** 0.5, data[:, sele] ** 1.5, data[:, sele] ** 2,data[:, sele] ** 2.5,data[:, sele] ** 3), axis = 1)
	return data

def test(test_data,Xtrain,Ytrain,num1,num0,mn_1,mn_0,std,ans_filename):

	ans_file = open(ans_filename, "w")
	ans_file.write("id,label\n")
	std_inv = np.linalg.pinv(std)
	w_t = np.dot((mn_1-mn_0),std_inv)
	b = (-0.5)*np.dot(np.dot(mn_1,std_inv),mn_1)+(0.5)*np.dot(np.dot(mn_0,std_inv),mn_0)+np.log(num1/num0)
	z = w_t.dot(test_data.T)+b
	prob = sigmoid(z)
	res_prob = np.around(prob)
	for i in range(len(res_prob)):
		ans_str = str(i+1) + ',' + str(int(res_prob[i])) + '\n'
		ans_file.write(ans_str)
		
trainX_name = sys.argv[1]
trainY_name = sys.argv[2]
testX_name = sys.argv[3]
ans_name = sys.argv[4]

(trainX_ret,trainY_ret,testX_ret) = read_data(trainX_name,trainY_name,testX_name)
#Xtrain = select(trainX_ret)
#Xtest = select(testX_ret)

(Xtrain_ret,Xtest_ret) = normalize(trainX_ret,testX_ret)

(num1,num0,mn_1,mn_0,std) = train(Xtrain_ret, trainY_ret)

test(Xtest_ret,Xtrain_ret, trainY_ret,num1,num0,mn_1,mn_0,std,ans_name)
