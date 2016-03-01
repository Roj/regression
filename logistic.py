import numpy as np
import cPickle, gzip
from scipy.special import expit

class LogisticRegression:
	def classify(self,theta,x):
		return expit(np.dot(theta,x))
	def cost(self,theta,samples):
	     total = 0
	     i = 0
	     for sample in samples:
	     	predicted = self.classify(theta,sample['vec'])
	     	if predicted > 0:   total += sample['class'] * np.log(predicted)
	     	if 1-predicted > 0: total += (1 - sample['class']) * np.log(1 - predicted)
	     return -total

	def gradient(self,theta,samples):
	     total = 0
	     for sample in samples:
	         total += sample['vec'] * (self.classify(theta,sample['vec']) - sample['class'])
	     return total

	def normalize(self,v):
		return v/np.linalg.norm(v)

	def right_wrong(self,theta,samples):
		right = 0
		wrong = 0
		for sample in samples:
			if round(self.classify(theta,sample['vec'])) == sample['class']:
				right += 1
			else:
				wrong += 1
		print "Right: " + str(right) + " Wrong: " + str(wrong)

	def train(self,samples,seed):
		theta = seed
		amount_iter = 20
		for it in range(1,amount_iter):
			grad = self.gradient(theta,samples)
			#print np.linalg.norm(grad)
			if np.linalg.norm(grad) != 0:
				grad = self.normalize(grad)
			theta = theta +  np.log(it*1.0/(amount_iter)) * grad
			if it %5 == 0:
				print "Iteration #" +str(it)+" current cost: " + str(self.cost(theta,samples))
			

		return theta
	def __init__(self,train_data,test_data):
		self.theta = np.random.rand(np.shape(samples[0]['vec'])[0]) # seed
		self.train_data = train_data
		self.test_data = test_data
	def run(self):
		print "Cost first: " + str(self.cost(self.theta,self.train_data))
		self.theta = self.train(samples,self.theta)
		print "Cost at the end: " + str(self.cost(self.theta,self.train_data))
		self.right_wrong(self.theta,self.train_data)

		print "So this is all great doctor, but how does it handle real data?"

		self.right_wrong(self.theta,self.test_data)



print "Loading data..."
#get the data at http://deeplearning.net/tutorial/gettingstarted.html
f = gzip.open("C:\Users\Joaquin\Downloads\mnist.pkl.gz","rb")
train_set,valid_set,test_set = cPickle.load(f)
f.close()

samples = []

for i in xrange(0,len(train_set[0])):
	if train_set[1][i] < 2: #zero or one
		samples.append({'vec': train_set[0][i], 'class': train_set[1][i]})


real_money = []
for i in xrange(0,len(test_set[0])):
	if test_set[1][i] < 2: #zero or one
		real_money.append({'vec': test_set[0][i], 'class': test_set[1][i]})

print "...loading DONE"
print
print


stanford = LogisticRegression(samples,real_money)
stanford.run()