import sys
import os
import numpy as np
import pandas as pd
import csv

np.random.seed(42)
NUM_FEATS = 90
class Net(object):
    def __init__(self,num_layers,num_units):
      self.temp_values={}
      self.weights=[]
      self.biases=[]
      for i in range(1,num_layers+2):
        if(i==1):
          size=(num_units,NUM_FEATS)
          
          self.weights.append(np.random.uniform(-1,1,size))
          
          size=(num_units,1)
          
          self.biases.append(np.random.uniform(-1,1,size))
        elif(i==(num_layers+1)):
          size=(1,num_units)
          
          self.weights.append(np.random.uniform(-1,1,size))
          size=(1,1)
          
          self.biases.append(np.random.uniform(-1,1,size))
        else:
          size=(num_units,num_units)
          
          self.weights.append(np.random.uniform(-1,1,size))
          size=(num_units,1)
          
          self.biases.append(np.random.uniform(-1,1,size))
        
        #print("w"+str(i)+" shape= ",np.array(self.weights[i-1]).shape," b"+str(i)+" shape= ",np.array(self.biases[i-1]).shape)
    
    def relu(x):
      return np.maximum(0,x)

    def __call__(self,X):
      no_of_layers = len(self.weights)

      for i in range(1, no_of_layers+1):
        if(i==1):
            self.temp_values['z' + str(i)] = np.dot(self.weights[i-1], X) + self.biases[i-1]
            self.temp_values['a' + str(i)] = np.maximum(0,self.temp_values['z' + str(i)])
        else:
            self.temp_values['z' + str(i)] = np.dot(self.weights[i-1], self.temp_values['a' + str(i-1)]) + self.biases[i-1]
            if(i==no_of_layers):
              self.temp_values['a' + str(i)] = self.temp_values['z' + str(i)]
            else:
              self.temp_values['a' + str(i)] = np.maximum(0,self.temp_values['z' + str(i)])
      
      return self.temp_values['a'+ str(no_of_layers)]

    

    def backward(self,X,y,lamda):
      def d_relu(x):
        # this function computes the differentiation of relu
        x[x<=0]=0
        x[x>0]=1
        return x
      # this function is for back propagation

      layers=len(self.weights)
      m= len(X)
      grads = {}
      dw=[]
      db=[]
      for i in range(layers , 0,-1):
        if(i==layers):
          dA = (1/m)*(self.temp_values['a'+str(i)] - y)
          dZ = dA
        else:
          dA = np.dot(self.weights[i].T,dZ)
          dZ = np.multiply(dA,d_relu(self.temp_values['a'+str(i)]))
        if i==1:
            dw.append(1/m * np.dot(dZ, X.T))
            db.append(1/m * np.sum(dZ, axis=1, keepdims=True))
        else:
            dw.append(1/m * np.dot(dZ,self.temp_values['a' + str(i-1)].T))
            db.append(1/m * np.sum(dZ, axis=1, keepdims=True))

      return dw,db

class Optimizer(object):

  def __init__(self,learning_rate):
    self.lamda = learning_rate
  
  def step(self,weights,biases,delta_weights,delta_biases):
    layers=len(weights)
    weights_updated=[]
    biases_updated=[]
    for i in range(0,layers):

      weights[i]=weights[i] - self.lamda*delta_weights[layers-i-1]
      biases[i]=biases[i] - self.lamda*delta_biases[layers-i-1]

    return weights,biases


def loss_mse(y,y_hat):
  cost = (1/(2*len(y_hat))) * (np.sum(np.square(y_hat - y)))
  return cost

def loss_regularization(weights,biases):
  layers=len(weights)
  loss_sum=0
  for i in range(0,layers):
    loss_sum = loss_sum + np.sum(np.sum(np.square(weights[i]))+np.sum(np.square(biases[i])))
  return loss_sum

def loss_fn(y,y_hat,weights,biases,lamda):
  # y is true label and y_hat is predicted by model
  number_of_samples=(np.array(y_hat).shape)
  loss = loss_mse(y,y_hat) + lamda * loss_regularization(weights,biases)
  return loss

def rmse(y,y_hat):
  return np.sqrt(np.mean(np.square(y_hat-y)))

def train(net,optimizer,lamda,batch_size,max_epochs,train_input,train_target,dev_input,dev_target):
  layers=len(net.weights)
  loss=[]
  total=train_input.shape[0]
  for i in range(max_epochs):
    for j in range(0,total,batch_size):
      x=train_input[j:j+batch_size]
      y=train_target[j:j+batch_size]

      y_pred = net(x.T)
      
      cost = loss_fn(y,y_pred,net.weights,net.biases,lamda)
      loss.append(cost)
      dw,db=net.backward(x.T,y.T,lamda)
      
      net.weights,net.biases=optimizer.step(net.weights,net.biases,dw,db)
      
    print("loss at epoch ",i," is = ",cost)
    
  train_pred=net(train_input.T)

  train_loss=rmse(train_target,train_pred)
  print("train_loss= ",train_loss)
  dev_pred=net(dev_input.T)
  dev_loss=rmse(dev_target,dev_pred)
  
  print("dev_loss= ",dev_loss)


def read_data():
  path="/home/nitish/Desktop/cs725_fml/assignment_2/cs725-autumn-2020-programming-assignment-2/dataset/"
  train_data = pd.read_csv(path+"train.csv")
  train_input=np.array(train_data.iloc[:,1:])
  train_target=np.array(train_data.iloc[:,0])

  test_data=pd.read_csv(path+"test.csv")
  test_input=np.array(test_data.iloc[:,:])

  dev_data=pd.read_csv(path+"dev.csv")
  dev_input=np.array(dev_data.iloc[:,1:])
  dev_target=np.array(dev_data.iloc[:,0])
  return train_input, train_target, dev_input, dev_target, test_input

def get_test_data_predictions(net, test_input):
  y_pred = net(test_input.T)
  no_of_layers=len(net.weights)
  pred=[]
  for i in range(0,y_pred.shape[1]):
    pred.append(int(round(y_pred[0][i])))
  filename = 'submission.csv'
  y_pred=[]
  for i in pred:
    y_pred.append(float(i))

  submission = pd.DataFrame({'Predicted':y_pred})
  submission.index+=1
  submission.index= submission.index.astype(str) 

  submission.to_csv(filename,index_label='Id')
  submit_data = pd.read_csv("submission.csv")
  os.remove('submission.csv')
  submit_data['Id'] = submit_data['Id'].astype('float64')
  submission = pd.DataFrame({'Id':submit_data['Id'],'Predicted':y_pred})
  submission.index+=1
  filename = '203050069.csv'
  submission.to_csv(filename,index=False)
  return y_pred
  
def main():
  max_epochs = 50
  batch_size = 128
  learning_rate = 0.001
  num_layers = 1
  num_units = 128
  lamda = 0.01 # Regularization Parameter

  train_input, train_target, dev_input, dev_target, test_input = read_data()

  net = Net(num_layers, num_units)
  
  optimizer = Optimizer(learning_rate)

  train(net, optimizer, lamda, batch_size, max_epochs,train_input, train_target,dev_input, dev_target)
  y_test_pred=get_test_data_predictions(net, test_input)
  #print(y_test_pred)

if __name__ == '__main__':
  main()

