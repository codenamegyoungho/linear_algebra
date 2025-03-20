# -*- coding: utf-8 -*-
"""
written by rudgh99 
For more information, please contact : dhkscjsrudgh@gmail.com
"""
# excercise 1 
import numpy as np 
import matplotlib.pyplot as plt 
l1,l2,l3 = 1,2,3 
v1 = np.array([1,2,3,4]) 
v2 = np.array([3,4,5,6]) 
v3 = np.array([5,6,7,8]) 

scalar,vector = [l1,l2,l3] , [v1,v2,v3] 

tmp1 = l1 * v1 + l2 * v2 + l3 * v3  
answer = np.zeros(len(v1))
for s,v in zip(scalar,vector): 
    answer += s * v 
answer,tmp1

#%% excercise 2 (causing error ; because indexing mismatch)
import numpy as np 
import matplotlib.pyplot as plt 
l1,l2,l3,l4 = 1,2,3,4
v1 = np.array([1,2,3,4]) 
v2 = np.array([3,4,5,6]) 
v3 = np.array([5,6,7,8]) 

scalar = [l1,l2,l3,l4] 
vector = [v1,v2,v3] 
lincombo = np.zeros(len(vector))
for i in range(len(scalar)):
    lincombo += scalar[i] * vector[i] 
lincombo

#%% excercise 3
v1 = np.array([1,3]) 
V = [v1] 
xlim = [-4,4] 
scalar = np.random.uniform(low=xlim[0],high=xlim[1],size=100) 

plt.figure(figsize=(6,6)) 
for s in scalar:
    p = v1 * s 
    plt.plot(p[0],p[1],'ko')

#%% 
import plotly.graph_objects as go 

v1 = np.array([3,4,5]) 
v2 = np.array([5,6,7]) 
scalars = np.random.uniform(low = xlim[0],high=xlim[1],size=(100,2))  
points = np.zeros((100,3))
for i in range(len(scalars)):
    points[i,:] = scalars[i,0] * v1 + scalars[i,1] * v2 

fig = go.Figure(data = [go.Scatter3d(x = points[:,0],y = points[:,1], z = points[:,2],mode='markers')])
import plotly.offline as pyo
pyo.plot(fig)  # fig.show() 대신 사용




