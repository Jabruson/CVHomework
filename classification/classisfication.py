import numpy as np
import pickle
#标签为0的数据
x0=[]
y0=[]

with open('train_data.pkl','rb')as f:
    samples,labels=pickle.load(f)
    
for i in samples:
       x0.append(i[0])
       y0.append(i[1])
shape1=len(labels)
x0=np.array(x0)
y0=np.array(y0)
labels=np.array(labels)
#初始化w权重矩阵
k=1
b=1
step=0.000002
train_num=3000
def delta(x,y,k,b,shape1):
    t0=k*x+b-y
    t1=t0*x
    t2=t0*t0
    m1=0
    m2=0
    m3=0
    for i in range(shape1):
        m1+=t1[i]
        m2+=t2[i]
        m3+=t0[i]
    m2=np.sqrt(m2)
    result1=m1/m2
    result2=m3/m2
    return result1,result2
    
def learn(x,y,k,b,shape1):
    delta0=delta(x,y,k,b,shape1)
    k-=(delta0[0]*step)
    b-=(delta0[1]*step)
    return k,b
    
for i in range(train_num):
    m=learn(x0,y0,k,b,shape1)
    k=m[0]
    b=m[1]
print("k={}".format(k))
print("b={}".format(b))

with open("test_data.pkl",'rb')as f:
    sample,label=pickle.load(f)
shape2=len(label)
#测试数据集
y1=[]
x1=[]
for sam in sample:
    x1.append(sam[0])
    y1.append(sam[1])
x1=np.array(x1)
y1=np.array(y1)
y_predict=k*x1+b
num_right=0
for i in range(shape2):
    if(label[i]==0):
        if y_predict[i]>y1[i]:
            num_right+=1
    else:
        if y_predict[i]<y1[i]:
            num_right+=1
right_prop=num_right/shape2
print("right prop:{}".format(right_prop))
import matplotlib.pyplot as plt
for i,sample1 in enumerate(sample):
    plt.plot(sample1[0],sample1[1],'o'if label[i]else '^',mec='r'if label[i]else 'b',mfc='none',markersize=10)
t1=np.linspace(0,10)
plt.plot(t1,k*t1+b,'k--')
plt.show()

