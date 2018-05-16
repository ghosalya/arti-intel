import numpy as np
import matplotlib.pyplot as mpl

def datagen(n):
  c=np.random.randint(1,3,size=(n))
  us=np.random.normal(loc=0,scale=1,size=(2,n))

  
  #######################
  # your code here
  a_pre = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                    [-np.sin(np.pi/4), np.cos(np.pi/4)]])
  a_post = np.array([[3,0],
                     [0,1]])

  mu1 = 0
  mu2 = 2.5

  A = np.matmul(a_pre, a_post)
  xs = np.matmul(A,us)
  ys = np.zeros((2,))

  for i in range(n):
    if c[i] == 1: 
      xs[i] += mu1
    elif c[i] == 2:
      xs[i] += mu2



  #######################
  
  return xs,ys,c
  
def plot1(n):
  xs,ys,c=datagen(n)

  #######################
  # your code here
  

  #######################
  

  
def knn(traindata, trainlabels,x):
  
  dists=np.ones(traindata.shape[1])
  minind=0

  #######################
  # your code here
  

  #######################  

      
  return trainlabels[minind]

def cls(n):
  xs,ys,c=datagen(n)
  
  xt,yt,ct=datagen(n)
  

  err0=0
  #######################
  # KNN prediction of training data
  # your code here
  

  #######################
  print ('err0',err0)

  err1=0
  #######################
  # KNN prediction of testing data
  # your code here
  

  #######################
  print ('err1',err1)
  
  err2=0
  #######################
  # other prediction of testing data
  # your code here
  

  #######################
  print ('err2',err2)  

  
if __name__=='__main__':
  #plot1(1000)
  cls(1000)

