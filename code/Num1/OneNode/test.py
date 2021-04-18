from vec import vecConv
import numpy as np
import time

channel=[128,64,32,16,8,4,2,1]
np.random.seed(1)

for i in channel:
    x=np.random.randn(64,64,i)
    k=np.random.randn(3,3,i,256)
    tic = time.process_time()
    vecConv(x,k,{"pad":0,"stride":1})
    toc = time.process_time()
    print ("i = "+str(i)+" Computation time for conv= " + str(1000*(toc - tic)) + "ms")



