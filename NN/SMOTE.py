import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Smote:
    def __init__(self,samples,N=10,k=5):
        print(samples.shape)
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1

import pandas as pd
data = pd.read_csv("../test_algorithm.csv", usecols=[0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43],encoding='gb2312')
data.fillna(value=-1, inplace=True)
# label = pd.read_csv("../test_algorithm.csv", usecols=[46], encoding='gb2312')
# label = np.array(label)
X = np.array(data)
X = X[:, 1:40]
# a=np.array([[1,2,3],[4,5,6],[2,3,1],[2,1,2],[2,3,4],[2,3,4]])
s = Smote(X, N=300)

output = s.over_sampling()
save = pd.DataFrame(output)
save.to_csv("SMOTE.csv")
