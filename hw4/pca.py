import numpy as np
from skimage import io
from os import listdir
from os.path import isfile, join
import sys


def readdata(filename):
	img_name = [i for i in listdir(filename) if(i[0]!='.')]
	img = []
	for iind in img_name:
		s_img = io.imread(join(filename,iind))
		#print(s_img.shape)
		s_img = s_img.flatten()
		#print(s_img.shape)
		img.append(s_img)
	img=np.array(img)
	#print(img.shape)

	return img

filename=sys.argv[1]
recon_name=sys.argv[2]

img_data = readdata(filename)
#print("img_data :",img_data.shape)
mean = np.mean(img_data,axis=0)
#print("mean shape :",mean.shape)
img_data = img_data - mean

(U,sigma,V_T) =np.linalg.svd(img_data.T, full_matrices=False)

#print(U.shape,sigma.shape,V_T.shape)


#resonstruction

c=4

re_img = io.imread(join(filename,recon_name))
re_img = re_img.flatten()
re_img = re_img - mean
wei = np.dot(re_img,U[:,:c])
cons_img = np.dot(U[:,:c],wei)
#print(cons_img.shape)
cons_img = cons_img + mean

cons_img -= np.min(cons_img)
cons_img /= np.max(cons_img)
cons_img = (cons_img * 255).astype(np.uint8)
#cons_img = mean
cons_img = cons_img.reshape((600,600,3))
save_name = "reconstruction.png"
io.imsave(save_name, cons_img)






