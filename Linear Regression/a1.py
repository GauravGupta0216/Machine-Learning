import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
import math

#--------------------------GRADIENT DESCENT---------------------------------#
def gradient_descent(theta, alpha, itr, h, X_train, Y_train):
    cost = np.zeros(itr)
    m = X_train.shape[0]
    n = X_train.shape[1]
    #X_train = X_train.transpose()
    #print(m,n)
    for i in range(0,itr):
        theta[0] = theta[0] - (alpha/m) * sum(h - Y_train)
        for j in range(1,n):
            theta[j] = theta[j] - (alpha/m) * sum((h-Y_train) * X_train.transpose()[j])
        #theta = theta.reshape(1,n) 
        #X_train = X_train.transpose()
        h = hypothesis(theta, X_train)
        cost[i] = sum(np.square(h - Y_train)/(2*m))
    theta = theta.reshape(1,n)
    print(theta)
    #print(cost)
    return theta, cost

#---------------------------HYPOTHESIS FUNCTION-----------------------------#
def hypothesis(theta, X_train):
    m = X_train.shape[0]
    n = X_train.shape[1]
    theta = theta.reshape(1,n)
    #print(theta)
    h = np.ones((m,1))
    for i in range(0,X_train.shape[0]) :
        h[i] = float(np.matmul(theta, X_train[i]))
    h = h.reshape(X_train.shape[0])
    #print(h)
    return h

#----------------------------READING DATA FILE------------------------------#
df = np.loadtxt('airfoil_self_noise.dat')
x = df[:,:5]
y = df[:,5]
#print(x.shape[1])
#print(x.shape[0])

#-----------------------------FEATURE SCALING--------------------------------#
x_mean = np.ones(x.shape[1])
x_sd = np.ones(x.shape[1])
y_mean = np.mean(y)
y_sd = np.std(y)
for i in range(0,x.shape[1]):                           #x.shape[1] is no. of column in x
    x_mean[i] = np.mean(x[:,i])
    x_sd[i] = np.std(x[:,i])
    for j in range(0,x.shape[0]):                       #x.shape[0] is no. of rows in x
        x[j][i] = (x[j][i] - x_mean[i])/x_sd[i]
        #x[j][i] = (x[j][i] - x_mean[i])/(max(x[:,i]) - min(x[:,i]))

#for i in range(0,y.shape[0]):
#    y[i] = (y[i] - y_mean)/y_sd

alpha = 0.01
itr = 4000

#SPILITTING THE DATA
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=20)

#--------------------------------regression--------------------------------#
n = X_train.shape[1]
m = X_train.shape[0]
X_train = np.concatenate((np.ones((m,1)), X_train), axis = 1)
#print(x)
theta = np.zeros(n+1)
h = hypothesis(theta, X_train)
#print(h)
theta, cost = gradient_descent(theta,alpha,itr,h,X_train,Y_train)
print(cost)
print(theta)

#---------------------predict the test data--------------------------------#
g = X_test.shape[0]
Y_pred = np.ones(g)
for i in range(0,g):
    Y_pred[i] = theta[0][0] + theta[0][1]*X_test[i][0] + theta[0][2]*X_test[i][1] + theta[0][3]*X_test[i][2] + theta[0][4]*X_test[i][3] + theta[0][5]*X_test[i][4]
#print(theta.shape)

#----------------------------PERFORMANCE ANALYSIS-------------------------#
sse = 0
sst = 0
for i in range(0,g):
    sse = sse + (Y_pred[i] - Y_test[i])**2
    sst = sst + (Y_test[i] - Y_test.mean())**2
    
mse = sse/g
r2 = 1 - (sse/sst)
print("mse = ", mse)
print("rmse = ", math.sqrt(mse))
print("sse = ", sse)
print("sst = ", sst)
print("r2 = ", r2)
#print(Y_pred)
#print(Y_test)
#print(y_pred.shape)
plt.plot(Y_test,Y_pred)
plt.xlabel('y_test')
plt.ylabel('y_predicted')
#plt.show()

#cost = list(cost)
n_iterations = [x for x in range(1,itr+1)]
plt.plot(n_iterations, cost)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.show()
#print(theta)
print(cost)

#print(Y_test)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
