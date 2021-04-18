import numpy as np
np.random.seed(1)
image=np.random.randn(1, 256, 256, 3) #h256X256 image
W1=np.random.randn(3, 3, 3, 32)
b1=np.random.randn(1, 1, 1, 32)
W2=np.random.randn(3, 3, 32, 64)
b2=np.random.randn(1, 1, 1, 64)
W3=np.random.randn(3, 3, 64, 128)
b3=np.random.randn(1, 1, 1, 128)
hparameters1 = {"pad" : 129,"stride": 2}
hparameters3 = {"pad" : 65,"stride": 2}
hparameters2 = {"stride" : 2, "f": 2}
hparameters4 = {"stride" : 2, "f": 2}
hparameters5 = {"pad" : 32,"stride": 2}

np.random.seed(1)
W_test=np.random.randn(2,2,1,6)
b1_test=np.zeros((1,1,1,1))

####################################333
[128,64,32,16,8,4,2,1]
W1=np.random.randn(9,9,1,256)
W2=np.random.randn(9,9,2,256)
W4=np.random.randn(9,9,4,256)
W8=np.random.randn(9,9,8,256)
W16=np.random.randn(9,9,16,256)
W32=np.random.randn(9,9,32,256)
W64=np.random.randn(9,9,64,256)
W128=np.random.randn(9,9,128,256)
kernels={"W1":W1,"W2":W2,"W4":W4,"W8":W8,"W16":W16,"W32":W32,"W64":W64,"W128":W128}



