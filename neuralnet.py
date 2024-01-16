import csv 
import math 
import sys 
import numpy as np  
from random import *

with open(sys.argv[1]) as csvin: 
    csvin = csv.reader(csvin, delimiter =',') 
    csvmatrix_train = []
    for row in csvin: 
        row  = map(int, row)
        csvmatrix_train.append(row)   

hidden_units = int(sys.argv[7]) 
init_flag = int(sys.argv[8])  
epoch = int(sys.argv[6])  
gamma = float(sys.argv[9])

yi_vector = []
for i in range(0,len(csvmatrix_train)): 
    yi_vector.append(csvmatrix_train[i][0]) 
    csvmatrix_train[i][0] = 1
csvmatrix_train = np.array(csvmatrix_train)
yi_vector = np.array(yi_vector) 


with open(sys.argv[2]) as csvin: 
    csvin = csv.reader(csvin, delimiter =',') 
    csvmatrix_val = []
    for row in csvin: 
        row  = map(int, row)
        csvmatrix_val.append(row)   

yi_vector_val = []
for i in range(0,len(csvmatrix_val)): 
    yi_vector_val.append(csvmatrix_val[i][0]) 
    csvmatrix_val[i][0] = 1
csvmatrix_val = np.array(csvmatrix_val)
yi_vector_val = np.array(yi_vector_val) 


M = csvmatrix_train[0].size - 1
D = hidden_units
K = 10  

if(init_flag == 2): 
    alpha = np.zeros((D+1, M+1))
    beta = np.zeros((K, D+1)) 

if(init_flag == 1): 

    alpha = np.zeros((D+1, M))
    for i in range(0, D+1):  
        a = np.random.uniform(-0.1,0.1,M) 
        alpha[i] = a 
    alpha = np.insert(alpha,0,0, axis = 1) 
    alpha[0] = np.zeros(M+1)
    
    beta = np.zeros((K, D))
    for i in range(0, K):  
        b = np.random.uniform(-0.1,0.1,D) 
        beta[i] = b  
    beta = np.insert(beta,0,0, axis = 1) 



class Forward(object): 
    def __init__(self, alpha, beta, x_vector, yi): 
        self.alpha = alpha
        self.beta = beta 
        self.xvec = x_vector
        self.yi = yi 

        self.a = []
        self.z = [] 
        self.b = [] 
        self.y_hat = [] 
        self.J = None

    def linear_forward(self): 
        b = np.multiply(self.xvec, self.alpha)
        a = np.sum(b, axis =1)
        self.a = a 
    def sigmoid_forward(self):  
        z = np.divide(1, (1+np.exp(-self.a)))
        z[0] = 1
        self.z = z  
    def linear_forward2(self): 
        b = np.multiply(self.z, self.beta) 
        r = np.sum(b, axis = 1)
        self.b = r
    def softmax_forward(self): 
        z_exp = np.exp(self.b)
        denom = np.sum(z_exp) 
        self.y_hat = np.divide(z_exp,(denom)) 

    def cross_entropy_forward(self): 
        yhat = self.y_hat[self.yi] 
        J = math.log(yhat)
        self.J = J  



class Backward(object): 
    def __init__(self, alpha, beta, x_vector, yi, a, z, b, y_hat, J): 
        self.alpha = alpha
        self.beta = beta 
        self.xvec = x_vector
        self.yi = yi 

        self.a = a
        self.z = z 
        self.b = b 
        self.y_hat = y_hat 
        self.J = J  

        self.gyk = [] 
        self.gbk = []
        self.gB_kj = []
        self.gzj = []
        self.gaj = []
        self.ga_ji = []


    def cross_entropy_backwards(self): 
        K = self.beta.shape[0]
        one_hot = np.zeros(K-1)
        one_hot = np.insert(one_hot, self.yi, 1)
        gyk = np.divide(one_hot, self.y_hat)
        self.gyk = -gyk 

    def softmax_backwards(self): 
        K = self.beta.shape[0]
        one_hot = np.zeros(K-1)
        one_hot = np.insert(one_hot, self.yi, 1) 
        indicator_vec = np.subtract(one_hot,self.y_hat)
        constant = self.gyk[self.yi]*self.y_hat[self.yi] 
        gbk = np.multiply(constant, indicator_vec)
        self.gbk = gbk 
    
    def linear_backwards(self): 
        K = self.beta.shape[0]
        gbk_trans = np.transpose([self.gbk]) 
        z_mat = np.repeat([self.z], K, axis=0)
        gB_kj = np.multiply(gbk_trans, z_mat)
        gB_kj_trans = np.multiply(self.gbk, np.transpose(self.beta))
        gzj = np.sum(gB_kj_trans, axis=1)
        self.gB_kj = gB_kj  
        self.gzj = gzj 

    def sigmoid_backwards(self): 
        zj_vector = np.multiply(self.z, np.subtract(1,self.z))
        gaj = np.multiply(self.gzj, zj_vector)
        self.gaj = gaj 
    def linear_backwards2(self):
        j = self.alpha.shape[0]
        gaj_trans = np.transpose([self.gaj])
        alpha_shell = np.repeat([self.xvec],j, axis = 0)
        ga_ji = np.multiply(gaj_trans, alpha_shell)
        self.ga_ji = ga_ji 




def sgd(alpha, beta, x_dat, y_dat,x_val,y_val, gamma, epoch):
    metric_out = open(sys.argv[5],'w') 
    # ce_train_out = open(sys.argv[10],'w')
    # ce_val_out = open(sys.argv[11], 'w')
 
    for j in range(0, epoch): 
        for i in range(0, x_dat.shape[0]): 
            f = Forward(alpha, beta, x_dat[i], y_dat[i])
            f.linear_forward()
            f.sigmoid_forward()
            f.linear_forward2()
            f.softmax_forward()
            f.cross_entropy_forward() 
            b = Backward(alpha, beta, x_dat[i], y_dat[i], f.a, f.z, f.b, f.y_hat, f.J)
            b.cross_entropy_backwards()
            b.softmax_backwards()
            b.linear_backwards() 
            b.sigmoid_backwards()
            b.linear_backwards2() 
            alpha = np.subtract(alpha, (np.multiply(gamma, b.ga_ji)))
            beta = np.subtract(beta, (np.multiply(gamma, b.gB_kj)))

        tot_train = 0
        for i in range(0, x_dat.shape[0]):
            f = Forward(alpha, beta, x_dat[i], y_dat[i])
            f.linear_forward()
            f.sigmoid_forward()
            f.linear_forward2()
            f.softmax_forward()
            f.cross_entropy_forward()
            tot_train += f.J
        ce_train = -tot_train / x_dat.shape[0]  
        

        #ce_train_out.write(str(ce_train) + "\n")

        a = "epoch="+str(j+1)+ " crossentropy(train): "+ str(ce_train)
        metric_out.write(a + "\n")

        
        tot_val = 0
        for i in range(0, x_val.shape[0]):
            fv = Forward(alpha, beta, x_val[i], y_val[i])
            fv.linear_forward()
            fv.sigmoid_forward()
            fv.linear_forward2()
            fv.softmax_forward()
            fv.cross_entropy_forward()
            tot_val += fv.J
        ce_val = -tot_val / x_val.shape[0]  
        

        #ce_val_out.write(str(ce_val) + "\n")

        b= "epoch="+str(j+1)+ " crossentropy(validation): "+ str(ce_val)
        metric_out.write(b + "\n") 

    return(alpha, beta) 

a = sgd(alpha, beta, csvmatrix_train, yi_vector,csvmatrix_val, yi_vector_val, gamma, epoch)

def predictions_train(alpha, beta, x_dat, y_dat):  
    pred_out_train = open(sys.argv[3],'w') 
    metric_out = open(sys.argv[5],'a')
    labels = []
    for i in range(0, x_dat.shape[0]): 
        f = Forward(alpha, beta, x_dat[i], y_dat[i])
        f.linear_forward()
        f.sigmoid_forward()
        f.linear_forward2()
        f.softmax_forward() 
        max_prob_index = np.argmax(f.y_hat)  
        pred_out_train.write(str(max_prob_index) + "\n")
        labels.append(max_prob_index)  
    err_num = 0
    for i in range(0, x_dat.shape[0]): 
        if(labels[i] != y_dat[i]):
            err_num += 1 
    error = err_num/float(x_dat.shape[0])
    metric_out.write("error(train): " + str(error) + "\n")
    metric_out.close()
    pred_out_train.close()
    return(labels)  

def predictions_val(alpha, beta, x_dat, y_dat): 
    pred_out_val = open(sys.argv[4],'w')  
    metric_out = open(sys.argv[5],'a')
    labels = []
    for i in range(0, x_dat.shape[0]): 
        f = Forward(alpha, beta, x_dat[i], y_dat[i])
        f.linear_forward()
        f.sigmoid_forward()
        f.linear_forward2()
        f.softmax_forward() 
        max_prob_index = np.argmax(f.y_hat) 
        pred_out_val.write(str(max_prob_index) + "\n") 
        labels.append(max_prob_index)  
    err_num = 0
    for i in range(0, x_dat.shape[0]): 
        if(labels[i] != y_dat[i]):
            err_num += 1 
    error = err_num/float(x_dat.shape[0])
    metric_out.write("error(validation): " + str(error) + "\n")
    metric_out.close()
    pred_out_val.close()
    return(labels) 

labels_train = predictions_train(a[0], a[1], csvmatrix_train, yi_vector) 

labels_val = predictions_val(a[0], a[1], csvmatrix_val, yi_vector_val) 

