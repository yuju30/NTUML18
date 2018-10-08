import numpy as np
import pandas as pd
import skimage 
import sys
from sklearn.decomposition import PCA 
from sklearn import cluster


def loaddata(file_name):
	raw_data = np.load(file_name)
    #print(raw_data.shape)
	return raw_data
def readtest(filename):
	test = pd.read_csv(filename,header=0).as_matrix()
	test = np.array(test).astype('int')
    #print(test.shape)
	return test


file_name = sys.argv[1]
data = loaddata(file_name)
test_file = sys.argv[2]
test_data = readtest(test_file)

p_mod=PCA(n_components=400,whiten=True)  
dim_data=p_mod.fit_transform(data)  
#print(dim_data.shape)
k_m_model = cluster.KMeans(n_clusters=2)
kmin_fit = k_m_model.fit(dim_data) 
ans = kmin_fit.labels_
#print(ans.shape)



ans_file = open(sys.argv[3],"w")
ans_file.write("ID,Ans\n")
for i in range(len(test_data)):
	if(ans[test_data[i][1]]==ans[test_data[i][2]]):	
		ans_str = str(test_data[i][0]) + ',' + str(int(1)) + '\n'
	else:
		ans_str = str(test_data[i][0]) + ',' + str(int(0)) + '\n'
	ans_file.write(ans_str)
ans_file.close()
