# -*- coding: utf-8 -*-
"""
written by rudgh99 
For more information, please contact : dhkscjsrudgh@gmail.com
"""
import numpy as np 

v = np.array([[4,5,6]]) # 행벡터 
w = np.array([[10,20,30]]).T # 열벡터 
print(v)
print(w)
u = np.array([0,3,6]) 
vplusW = v + w 
uplusW = u + w 
#print(vplusW) 
print(uplusW)
#%%
s = 2 
v = np.array([1,2,3]) 
v1 = [1,2,3] 
print(v1 * s) 
#%%
a = np.array([1,2,3]) 
b = np.array([4,5,6]) 
c = np.array([7,8,9]) 
s = np.dot(a,b+c) 
s1 = np.dot(a,b) + np.dot(a,c) 
print(s1 - s)
#%% excercise1  
import numpy as np
import matplotlib.pyplot as plt 
v1 = np.array([1,2]) 
v2 = np.array([4,-6]) 
v1plusv2 = v1 + v2  

plt.figure(figsize=(6,6))
a1 = plt.arrow(0,0,v1[0],v1[1],head_width = .3 ,width=.1,color='k',length_includes_head=True)
a2 = plt.arrow(v1[0],v1[1],v2[0],v2[1],head_width=.3,width=.1,color=[.5,.5,.5],length_includes_head=True)
a3 = plt.arrow(0,0,v1plusv2[0],v1plusv2[1],head_width=.3,width=.1,color=[.8,.8,.8],length_includes_head=True)

plt.grid(linestyle = '--',linewidth=.5) 
plt.axis('Square')
plt.axis([-6,6,-6,6])
plt.legend([a1,a2,a3],['v1','v2','v1 + v2'])
plt.title("vectors summization ; v1 + v2 ")
plt.show()

#%% excercise 2 (vector norm) 
def normOfVector(v):
    return np.sqrt(np.sum(v**2)) 

a = np.array([1,2,3])
normOfVector(a) 
print(normOfVector(a),np.linalg.norm(a))

#%% excercise 3 (standard vector; The size of vector is 1) 
def vomitStandardVector(v):
    #v = v / normOfVector(v) or
    return v / np.linalg.norm(v)
a = np.array([1,2,3]) 
vomitStandardVector(a)

#%% excercise 4 (I wanna spit out the specific vector) 
def makingVector(v,input_size):
    return input_size * v 
a = np.array([1,2,3]) 
makingVector(a,3)

#%% excercise 5 (transpose vector) 
v1 = np.array([[1,2,3]])
vs = np.zeros((3,1)) 
for i in range(v1.shape[1]):
    vs[i,0] = v1[0,i] 
vs 
#%% excercise 6 (vector norm square) 
v1 = np.array([1,2,3]) 
print(np.dot(v1,v1))

#%% excercise 7 (exchange raw of vector) 
v1 = np.array([1,2,3]) 
v2 = np.array([4,5,6]) 
print(np.dot(v1,v2),np.dot(v2,v1))

#%% excercise 8 () 
v1 = np.array([1,2]) 
v2 = np.array([1.5,0.5]) 
beta = np.dot(v1,v2) / np.dot(v1,v1) 

projectv = v2 - beta * v1 

plt.figure(figsize = (4,4))

plt.arrow(0,0,v1[0],v1[1],head_width=.2,width=.02,color="k",length_includes_head=True) 
plt.arrow(0,0,v2[0],v2[1],head_width=.2, width=.02, color="k",length_includes_head=True) 
# projection vector
plt.plot([v2[0],beta * v1[0]],[v2[1],beta*v1[1]],'k--') 
plt.plot(beta * v1[0], beta * v1[1], 'ko',markerfacecolor='w',markersize=13) 

# make the plot look nicer
plt.plot([-1,2.5],[0,0],'--',color='gray',linewidth=.5)
plt.plot([0,0],[-1,2.5],'--',color='gray',linewidth=.5)
# make the plot look nicer
plt.plot([-1,2.5],[0,0],'--',color='gray',linewidth=.5)
plt.plot([0,0],[-1,2.5],'--',color='gray',linewidth=.5) 
# add labels
plt.text(v1[0]+.1,v1[1],'a',fontweight='bold',fontsize=18)
plt.text(v2[0],v2[1]-.3,'b',fontweight='bold',fontsize=18)
plt.text(beta*v1[0]-.35,beta*v1[1],r'',fontweight='bold',fontsize=18)
plt.text((v2[0]+beta*v1[0])/2,(v2[1]+beta*v1[1])/2+.1,r'(v2-v1)',fontweight='bold',fontsize=18)

# some finishing touches
plt.axis('square')
plt.axis([-1,2.5,-1,2.5])
plt.show() 

#%% excercise 9 
t = np.random.randn(2) 
r = np.random.randn(2) 

beta = np.dot(r,t) / np.dot(r,r) 
betav = r * beta 
betavminus = t - betav 

plt.figure(figsize = (8,8))  
plt.arrow(0,0,t[0],t[1],color="k",head_width=0.2,width=0.02,length_includes_head=True)
plt.arrow(0,0,betav[0],betav[1],color="k",head_width=0.2,width=.02,length_includes_head=True)
plt.arrow(0,0,betavminus[0],betavminus[1],color="k",head_width=.2,width=0.02,length_includes_head=True) 


#%% excercise 10 
# skip 









