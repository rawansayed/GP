import numpy as np 
a = np.random.randint(2,size=(8,23))
print(a)
l=[]
for i in range( 1000):
    a =np.random.randint(2,size=(8,23))
    l.append(a)
np.save('./TrainingAutoEncoder/a.npy', l, allow_pickle=True)
