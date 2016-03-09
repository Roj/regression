# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 11:28:13 2016

@author: jtorre
"""
import math
import numpy as np
import cPickle, gzip
import random
#import scipy


class SoftmaxRegression:
    minibatch_size = 30
    iterations = 100
    def __init__(self,train,test,k):
        self.train_data = train
        self.test_data = test
        self.theta = np.random.rand(len(train[0]['vec']),k)    
    #denominator used in cost and classification
    def denominator(self,theta,X_i): 
        total = 0
        for i in xrange(0,theta.shape[1]):
            exponent = np.dot(theta[:,i],X_i)
            total += math.exp(exponent - 50)
        return total
    #Cost is used with all data
    def cost(self,theta,samples):
        #Theta is an n-by-k matrix
        total = 0
        for i in xrange(0,len(samples)):
            for j in xrange(0,theta.shape[1]):
                if samples[i]['class'] == j:
                        term = math.exp(np.dot(theta[:,j],samples[i]['vec']) - 50)
                        bottom = self.denominator(theta,samples[i]['vec']);
                        term = term/bottom
                        total+= np.log(term)
        return -total
    def single_numerator(self,theta,k,X_i): #return the k-numerator of the classify
        exponent = np.dot(theta[:,k],X_i)        
        return math.exp(exponent - 50)
    #mini batch gradient descent
    def single_gradient(self,theta,k,samples):
        
        total = 0
        begin = int(math.floor(random.random() * len(samples)))
        end = min(begin+self.minibatch_size,len(samples))
        for i in xrange(begin,end):
            sample = samples[i]
            special_term = 1 if sample['class'] == k else 0
            bottom = self.denominator(theta,sample['vec'])
            total -= sample['vec'] * (special_term - self.single_numerator(theta,k,sample['vec']) * 1.0/bottom)
        return total
    #Gradient is used with a subset
    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm >0: return v * 1.0/norm
        return v
    def fix_theta(self):
        max_norm = 0
        max_column = 0
        for j in xrange(0,self.theta.shape[1]):
            norm = np.linalg.norm(self.theta[:,j])
            if norm > max_norm:
                max_norm = norm
                max_column = j
        #we can do this beacause softmax is over-parameterized
        for j in xrange(0,self.theta.shape[1]):
            self.theta[:,j] = self.theta[:,j] - self.theta[:,max_column]
            
    def update_theta(self):
        old_theta = self.theta
        for i in xrange(0,self.theta.shape[1]):
            grad = self.single_gradient(old_theta,i,self.train_data)
            self.theta[:,i] = old_theta[:,i] - self.normalize(grad)
        #self.fix_theta()
        
    def train(self):
        amount_iter = self.iterations
        for i in xrange(0,amount_iter):
            self.update_theta()
            if i%10 == 0: 
                print "Iteration #" + str(i)+ ": current cost is "+str(self.cost(self.theta,self.train_data))
    def run(self):
        print "Cost at the beginning: "+str(self.cost(self.theta,self.train_data))
        self.train()
        print "Cost at the end: "+str(self.cost(self.theta,self.train_data))
    def classify(self,X_i):
        max = 0
        max_class = 0
        #No need for the denominator
        for j in xrange(0,self.theta.shape[1]):
            numerator = self.single_numerator(self.theta,j,X_i)
            if numerator > max:
                max_class = j
                max = numerator
        return max_class
    def right_wrong(self,samples):
        right = 0
        wrong = 0
        for sample in samples:
            if self.classify(sample['vec']) == sample['class']:
                right += 1
            else:
                wrong += 1
        print "Right: " + str(right) + " Wrong: " + str(wrong)
        print "Correct: " + str(right* 100.0 /(right+wrong)) + "%"

print "Loading data..."
#get the data at http://deeplearning.net/tutorial/gettingstarted.html
f = gzip.open("C:/Users/jtorre/Desktop/regression/mnist.pkl.gz","rb")
train_set,valid_set,test_set = cPickle.load(f)
f.close()

samples = []

for i in xrange(0,len(train_set[0])):
    samples.append({'vec': train_set[0][i], 'class': train_set[1][i]})


real_money = []
for i in xrange(0,len(test_set[0])):
    real_money.append({'vec': test_set[0][i], 'class': test_set[1][i]})

print "...loading DONE"
print
print


stanford = SoftmaxRegression(samples,real_money,10)
stanford.run()
stanford.right_wrong(stanford.train_data) # around 65-80%