import numpy as np
import pickle
samples=[]
labels=[]
samples_test=[]
labels_test=[]
np.random.seed(50)
def func(x):
    return 0.7*x+0.6
#训练集
f="train_data.pkl"
for i in range(1000):
    x1,x2=np.random.uniform(0,10,2)
    y=func(x1)
    if (x2-0.3)<y<(x2+0.3):
        continue
    else:
        samples.append((x1,x2))
        if x2>y:
            labels.append(1)
        else:
            labels.append(0)
with open(f,'wb')as data:
    pickle.dump((samples,labels),data)
np.random.seed(150)
#测试集
f2="test_data.pkl"
for i in range(50):
    x11,x22=np.random.uniform(0,10,2)
    y=func(x11)
    if (x22-0.1)<y<(x22+0.1):
        continue
    else:
        samples_test.append((x11,x22))
        if x22>y:
            labels_test.append(1)
        else:
            labels_test.append(0)
with open(f2,'wb')as data:
    pickle.dump((samples_test,labels_test),data)


#将数据可视化
#import matplotlib.pyplot as plt
#for i,sample in enumerate(samples):
#    plt.plot(sample[0],sample[1],'o'if labels[i]else '^',mec='r'if labels[i]else 'b',mfc='none',markersize=10)
#x1=np.linspace(0,10)
#plt.plot(x1,func(x1),'k--')
#plt.show()
    
