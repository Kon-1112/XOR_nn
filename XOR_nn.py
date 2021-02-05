import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

sigmoid_range = 34.538776394910684
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x,-sigmoid_range,sigmoid_range)))

def derivative_sigmoid(o):
    return o * (1.0 - o)

class ThreeLayerNetwork:
    
    def __init__(self,inodes,hnodes,onodes,lr):
        
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        
        self.lr = lr
        
        self.w_ih = np.random.normal(0.0,0.2,(self.hnodes,self.inodes))
        self.w_ho = np.random.normal(0.0,0.2,(self.onodes,self.hnodes))
        
        self.af = sigmoid
        self.daf = derivative_sigmoid
    
    def backprop(self,idata,tdata):
        o_i = np.array(idata,ndmin=2).T
        t = np.array(tdata,ndmin=2).T
        x_h = np.dot(self.w_ih,o_i)
        o_h = self.af(x_h)
        x_o = np.dot(self.w_ho,o_h)
        o_o = self.af(x_o)
        e_o = (t - o_o)
        e_h = np.dot(self.w_ho.T,e_o)
        
        self.w_ho += self.lr * np.dot((e_o * self.daf(o_o)),o_h.T)
        self.w_ih += self.lr * np.dot((e_h * self.daf(o_h)),o_i.T)
        
    
    def feedforward(self,idata):
        o_i = np.array(idata,ndmin=2).T
        x_h = np.dot(self.w_ih,o_i)
        o_h = self.af(x_h)
        x_o = np.dot(self.w_ho,o_h)
        o_o = self.af(x_o)
        return o_o


inodes = 2
hnodes = 10
onodes = 1
lr = 0.8


nn = ThreeLayerNetwork(inodes,hnodes,onodes,lr)


training_data_list = ['0,0,0','0,1,1','1,0,1','1,1,0']


training_time = 10000
e = np.zeros(2)
for t in range(training_time):
    data_size = len(training_data_list)
    error = 0
    if t % 1000 == 0:
        print('Learning time ',t,'/',training_time)
    
    
    for i in range(data_size):
        val = training_data_list[i].split(',')
        idata = np.asfarray(val[0:2])
        tdata = np.asfarray(val[2])
        
        error += pow(nn.feedforward(idata)-tdata,2)
        
        nn.backprop(idata,tdata)
        
        pass
    pass


    if t == 1:
        e = [t,error]
    else:
        e = np.vstack((e,[t,error]))


print('学習結果')
data_size = len(training_data_list)
for i in range(data_size):
    val = training_data_list[i].split(',')
    idata = np.asfarray(val[0:2])
    tlabel = np.asfarray(val[2])
    
    predict = nn.feedforward(idata)
    print(idata,'>',predict)
    pass

fig = plt.figure(figsize=(12,5))
gl = fig.add_subplot(1,2,1)
gl.plot(e[:,0],e[:,1])
plt.xlabel("Learning time")
plt.ylabel("Error")

g2 = fig.add_subplot(1,2,2)
O = np.array([0.1,0.1,float(nn.feedforward((0.1,0.1)))])
for x in range(-5,15,1):
    for y in range(-5,15,1):
        k = np.array([0.1*x,0.1*y,float(nn.feedforward((0.1*x,0.1*y)))])
        O = np.vstack((O,k))
s = g2.scatter(O[:,0],O[:,1],vmin=0,vmax=1,c=O[:,2],cmap='Blues')
plt.colorbar(s)
plt.show()    