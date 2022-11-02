import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90

class Net:

	def __init__(self, n_inputs, n_neurons,lamda=0):

		self.weights = 0.01*np.random.uniform(-1,1,(n_inputs,n_neurons))
		self.biases = np.zeros((1,n_neurons))
		self.lamda=lamda

	def forward(self,inputs):
		self.inputs=inputs	
		self.output = np.dot(inputs,self.weights) + self.biases
		
	def backward(self,dvalues):
		
		self.dweights=np.dot(self.inputs.T,dvalues)
		self.dbiases=np.sum(dvalues,axis=0,keepdims=True)

		if self.lamda>0:
			self.dweights+=2*self.lamda*self.weights
			self.dbiases+=2*self.lamda*self.biases

		self.dinputs=np.dot(dvalues,self.weights.T)

# Dropout Layer for Regularization
class Dropout:
	def __init__(self,rate):
		self.rate=1-rate

	def forward(self,inputs):
		self.inputs=inputs
		self.mask=np.random.binomial(1,self.rate,size=inputs.shape)/self.rate
		self.output=inputs*self.mask

	def backward(self,dvalues):
		self.dinputs=dvalues*self.mask

# ReLU activation for Hidden Layer
class ReLU:
	def forward(self,inputs):
		self.inputs=inputs
		self.output = np.maximum(0,inputs) 

	def backward(self,dvalues):
		self.dinputs=dvalues.copy()
		self.dinputs[self.inputs<=0]=0

# Linear Output Layer for Regression
class Linear_Output:
	def forward(self,inputs):
		self.inputs=inputs
		self.output=inputs
	def backward(self, dvalues):
		self.dinputs=dvalues.copy()

# Stochoastic Gradient Descent Algorithm
class SGD:
	def __init__(self,lr=1., decay=0.):
		self.lr=lr
		self.current_lr = lr
		self.decay=decay
		self.iterations=0

	def update_lr(self):
		if self.lr:
			self.current_lr=self.lr * (1./(1. + self.decay*self.iterations))

	def update_parameters(self,layer):
		layer.weights += -self.current_lr*layer.dweights
		layer.biases += -self.current_lr*layer.dbiases

	def inc_iterations(self):
		self.iterations +=1

# Adaptive Moment Estimation Algorithm
class Adam:
	def __init__(self,lr=0.001, decay=0., e=1e-7, m1=0.9, m2=0.999):
		self.lr=lr
		self.current_lr = lr
		self.decay=decay
		self.iterations=0
		self.e=e
		self.m1=m1
		self.m2=m2

	def update_lr(self):
		if self.decay:
			self.current_lr=self.lr * (1./(1. + self.decay*self.iterations))

	def update_parameters(self,layer):
	
		if not hasattr(layer,'weight_cache'):
			layer.weight_momentums=np.zeros_like(layer.weights)
			layer.weight_cache=np.zeros_like(layer.weights)
			layer.bias_momentums=np.zeros_like(layer.biases)
			layer.bias_cache=np.zeros_like(layer.biases)

		layer.weight_momentums=self.m1*layer.weight_momentums+(1-self.m1)*layer.dweights
		layer.bias_momentums=self.m1*layer.bias_momentums+(1-self.m1)*layer.dbiases

		weight_momentums_corected=layer.weight_momentums/(1-self.m1**(self.iterations+1))
		bias_momentums_corrected=layer.bias_momentums/(1-self.m1**(self.iterations+1))

		layer.weight_cache=self.m2*layer.weight_cache+(1-self.m2)*layer.dweights**2
		layer.bias_cache=self.m2*layer.bias_cache+(1-self.m2)*layer.dbiases**2

		weight_cache_corrected=layer.weight_cache/(1-self.m2**(self.iterations+1))
		bias_cache_corrected=layer.bias_cache/(1-self.m2**(self.iterations+1))

		layer.weights += -self.current_lr*weight_momentums_corected/(np.sqrt(weight_cache_corrected)+self.e)
		layer.biases += -self.current_lr*bias_momentums_corrected/(np.sqrt(bias_cache_corrected)+self.e)

	# after upadate
	def inc_iterations(self):
		self.iterations +=1

# General Loss Class for different regularized loss & other loss functions
class Loss:
	def reg_loss(self,layer):
		reg_loss=0
		if layer.lamda>0:
			reg_loss+=layer.lamda*np.sum(layer.weights*layer.weights)
			reg_loss+=layer.lamda*np.sum(layer.biases*layer.biases)
		return reg_loss
		
	def calculate(self,output,y):
		sample_losses= self.forward(output,y)
		data_loss=np.mean(sample_losses)
		return data_loss

# Categorical Cross Entropy Loss
class Cross_Entropy_Loss(Loss):
	def forward(self,y_pred,y_true):
		samples=len(y_true)
		y_pred_clipped= np.clip(y_pred,1e-7,1-1e-7)

		if len(y_true.shape)==1:
			correct_probability=y_pred_clipped[range(samples),y_true]
		elif len(y_true.shape)==2:
			correct_probability=np.sum(y_pred_clipped*y_true,axis=1)
		
		return(-np.log(correct_probability))

	def backward(self,dvalues,y_true):
		samples=len(dvalues)
		labels=len(dvalues[0])

		if len(y_true.shape)==1:
			y_true=np.eye(labels)[y_true]

		self.dinputs=-y_true/dvalues
		self.dinputs=self.dinputs/samples

# Mean Square Loss
class Loss_MSE(Loss):
	def forward(self,y_true,y_pred):
		sample_losses=np.mean((y_true-y_pred)**2, axis=-1)
		return sample_losses
	def backward(self,dvalues,y_true):
		samples=len(dvalues)
		outputs=len(dvalues[0])
		self.dinputs=-2*(y_true-dvalues)/outputs
		self.dinputs=self.dinputs/samples

# Root Mean Square Loss
class Loss_RMSE(Loss):
	def forward(self,y_true,y_pred):
		sample_losses_mse=np.mean((y_true-y_pred)**2, axis=-1)
		sample_losses=np.sqrt(sample_losses_mse)
		return sample_losses

# Accuracy for classification task
class Accuracy: 
	def calculate(self,output,y):
		pred=np.argmax(output,axis=1)

		if len(y.shape)==1:
			accuracy=np.mean(pred==y)

		elif len(y.shape)==2:
			tar=np.argmax(y,axis=1)
			accuracy=np.mean(pred==tar)

		return accuracy

def read_data():

	# trainig data
	df1 = pd.read_csv('regression/train.csv')
	#print(df.head(5))

	#input features
	train_input= df1.iloc[:,1:].values
	#print(X)
	#normalization
	#train_input = (train_input - train_input.mean())/train_input.std()
	
	#scaling bw -1 & 1
	train_input = (train_input/np.max(np.abs(train_input)))
	#print(X)
	train_target = df1.iloc[:,0].values
	train_target= np.expand_dims(train_target, axis=1)

	#validation data	
	df2 = pd.read_csv('regression/dev.csv')
	#print(df.head(5))

	#input features
	dev_input= df2.iloc[:,1:].values
	#print(X)
	#normalization
	#dev_input = (dev_input - dev_input.mean())/dev_input.std()
	
	#scaling bw -1 & 1 
	dev_input = (dev_input/np.max(np.abs(dev_input)))
	#print(X)
	dev_target = df2.iloc[:,0].values
	dev_target= np.expand_dims(dev_target, axis=1)


	df3 = pd.read_csv('regression/test.csv')
	#print(df.head(5))
	#input features
	test_input= df3.iloc[:,:].values
	#print(X)
	#normalization
	#test_input = (test_input - test_input.mean())/test_input.std()
	
	#scaling bw -1 & 1
	test_input = (test_input/np.max(np.abs(test_input)))
	#print(test_input)


	return train_input, train_target, dev_input, dev_target, test_input

def train(FCL1,A1,D12,FCL2,A2,D23,FCL3,A3,lamda,Loss_Fn,optimizer,max_epochs,batch_size,train_input,train_target):
	
	#Hyperparameters for Early Stopping
	minloss=9999999999
	patience=50 #epochs
	count=0

	precision=np.std(train_target)

	for epoch in range(1001):

		FCL1.forward(train_input)
		A1.forward(FCL1.output)

		D12.forward(A1.output)

		FCL2.forward(D12.output)
		A2.forward(FCL2.output)

		D23.forward(A2.output)

		FCL3.forward(D23.output)
		A3.forward(FCL3.output)

		data_loss=Loss_Fn.calculate(A3.output,train_target)
		reg_loss=Loss_Fn.reg_loss(FCL1)+Loss_Fn.reg_loss(FCL2)+Loss_Fn.reg_loss(FCL3)

		loss=data_loss+reg_loss

		predictions=np.round(np.squeeze(A3.output))

		#comment for faster execution
		#accuracy=np.mean(np.absolute(predictions-train_target)<precision)
		
		print(	f'epoch: {epoch},' + 
				#f'accuracy: {accuracy:.3f},' + 
				f'loss:{loss:.3f}, '+
				f'data_loss:{data_loss:.3f}, '+
				f'regularization_loss:{reg_loss:.3f},' + 
				f'lr:{optimizer.current_lr}')

		#early stopping criteria with patience of 25 epochs
		#without a checkpointer on weights
		'''
		if loss>minloss:
			count +=1
			if count>patience:
				print("Early Stopping")
				plt.show()
				return
		else:
			minloss=loss
		'''

		Loss_Fn.backward(A3.output,train_target)
		
		A3.backward(Loss_Fn.dinputs)
		FCL3.backward(A3.dinputs)
		
		D23.backward(FCL3.dinputs)
		
		A2.backward(D23.dinputs)
		FCL2.backward(A2.dinputs)
		
		D12.backward(FCL2.dinputs)
		
		A1.backward(D12.dinputs)
		FCL1.backward(A1.dinputs)

		optimizer.update_lr()
		optimizer.update_parameters(FCL1)
		optimizer.update_parameters(FCL2)
		optimizer.update_parameters(FCL3)
		optimizer.inc_iterations()

		plt.scatter(epoch,loss)
	plt.show()

def get_test_data_predictions(FCL1,A1,D12,FCL2,A2,D23,FCL3,A3,test_input):

	FCL1.forward(test_input)
	A1.forward(FCL1.output)

	D12.forward(A1.output)

	FCL2.forward(D12.output)
	A2.forward(FCL2.output)

	D23.forward(A2.output)

	FCL3.forward(D23.output)
	A3.forward(FCL3.output)
	
	predictions=np.round(np.squeeze(A3.output))

	return predictions

def main():

	max_epochs = 1501
	batch_size = 256
	lr = 0.1
	lamda = 0.0001
	e=1e-8
	decay=0.0001
	m1=0.9
	m2=0.999
	#loss=99.902
		
		
	train_input, train_target, dev_input, dev_target, test_input = read_data()

	FCL1=Net(90,64,lamda)
	A1=ReLU()

	D12=Dropout(0)

	FCL2=Net(64,32)
	A2=ReLU()

	D23=Dropout(0)
	
	FCL3=Net(32,1)
	A3=Linear_Output()


	Loss_Fn=Loss_MSE()

	optimizer=Adam(lr,e,decay,m1,m2)
	#optimizer=SGD(lr,decay)
	
	train(
		FCL1,A1,D12,FCL2,A2,D23,FCL3,A3,lamda,
		Loss_Fn,optimizer,max_epochs,batch_size,
		train_input,train_target
		)

	predictions=get_test_data_predictions(FCL1,A1,D12,FCL2,A2,D23,FCL3,A3,test_input)
	df=pd.DataFrame(predictions)
	df.to_csv(r'Regression/Result.csv')
 	

if __name__ == '__main__':
	main()
