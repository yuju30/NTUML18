import numpy as np
import pandas as pd
import sys

def Read_train(filename):
	ori_train_data = pd.read_csv(filename,encoding="big5").as_matrix()
	train_data = ori_train_data[:,3:]
	#12(months)*20(days)*18(type)
	train_data[train_data=='NR'] = 0.0
	train_data = train_data.astype(float)
	train_data[train_data<=0] = 0.0
	(y,x) = train_data.shape # 12*20*18,24	
	al = []
	pm_ans = []
	for i in range(0,y,360):
		days = np.vsplit(train_data[i:i+360],20) # 20 days * [18(types)*24(hours)]
		months = np.concatenate(days,axis=1) #18(types)*[24(hours)*20(days)]
		for v in range(0,18):
			for h in range(0,480):
				if(months[v][h]==0 and h>0):
					k=h+1
					while k < 480 and months[v][k]==0:
						k=k+1
					if(k==480):
						months[v][h]=months[v][h-1]
					else:
						months[v][h]=(months[v][h-1]+months[v][k])/2
		for j in range(0,months.shape[1]-9): #per 9 hours
			tp_blo_tp = np.concatenate((months[6:7,j:j+9],months[9:12,j:j+9]))
			tp_blo = np.concatenate((tp_blo_tp,months[7:8,j:j+9]*months[2:3,j:j+9]))
			
			al.append(tp_blo.flatten())
			pm_ans.append(months[9,j+9])
	al = np.array(al) #[471*12]*18*9

	pm_ans = np.array(pm_ans)#[471*12]
	return al,pm_ans

def Read_test(filename):
	ori_test_data = pd.read_csv(filename,encoding="big5",header = None).as_matrix()
	test_data=ori_test_data[:,2:]
	test_data[test_data=='NR']=0.0
	test_data=test_data.astype(float)
	id_num=int(test_data.shape[0]/18)
	testblock=np.vsplit(test_data,id_num) # 18 (rows) blocks * id_num 
	testblock=np.array(testblock)
	return testblock


def train(train_al,train_ans):
	b = -0.1
	lr = 0.5
	w  = np.array([0.5]*5*9)
	len_pm=len(train_al)
	sgra = np.array([0.1]*5*9)
	sgra_b = 0.1

	Last_RSE = 0.1
	g=10000
	while (g>0):
		g=g-1
		dLw = np.array([0.0]*5*9)
		dLb = 0.0
		
		tpres = np.dot(train_al,np.transpose(w))
		
		loss = (train_ans-(tpres+b))
	
		cost = ((loss**2).sum()/len_pm)
		
		dLw = dLw - (2.0*np.dot(loss,train_al))/len_pm
		dLb = dLb - (2.0*(loss.sum()))/len_pm

		RSE_pm = (cost)**(0.5)
		print(RSE_pm)
		if(abs(RSE_pm-Last_RSE)< 5*1e-6):
			break
		Last_RSE = RSE_pm
		#adagrad
		sgra = sgra + dLw**2
		sgra_b = sgra_b + dLb**2

		w = w - (lr*dLw)/np.sqrt(sgra)
		b = b - (lr*dLb)/np.sqrt(sgra_b)
	return w ,b


def test(test_data,w,b,ans_filename):
	ans_file = open(ans_filename, "w")
	ans_file.write("id,value\n")
	len_test=len(test_data)
	for i in range(len_test):
		cut_test_tp = np.concatenate((test_data[i,6:7,:], test_data[i,9:12,:]))
		cut_test = np.concatenate((cut_test_tp, test_data[i,7:8,:]*test_data[i,2:3,:]))
		cut_test=cut_test.flatten()
		pre_res = np.dot(cut_test,np.transpose(w))
		ans_str = "id_" + str(i) + "," +  str(pre_res) + "\n"
		ans_file.write(ans_str)
#train_csv=sys.argv[1]
test_csv=sys.argv[1]
ans_csv=sys.argv[2]
al_res=[]
pm_ans=[]
testblock_res=[]
#(al_res,pm_ans)=Read_train(train_csv)
(testblock_res)=Read_test(test_csv)
#(w_res,b_res) = train(al_res,pm_ans)
# save model
#np.save('model.npy',(w_res,b_res))
# read model
(w_res,b_res) = np.load('model.npy')
test(testblock_res,w_res,b_res,ans_csv)