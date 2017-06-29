import pandas as pd
import numpy as np
from sklearn.utils import shuffle
data = pd.read_csv("../TrainData_temp.csv",usecols=[0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43, 46],encoding='gb2312')
data.fillna(value=-1, inplace=True)
label = pd.read_csv("../TrainData_temp.csv", usecols=[46], encoding='gb2312')
label = np.array(label)
X = np.array(data)
X = X[:, 1:41]

# print(X)
# print("#######################################################################")
df = pd.DataFrame(X)
# df.sample(frac=1)
df = shuffle(df)

df.to_csv("Shuffled.csv")
# print(df)
