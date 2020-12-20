import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
np.random.seed(42)

def get_features(csv_path,is_train=False,scaler=None):
    train_data = pd.read_csv(csv_path)
    arr=np.array(train_data.iloc[:,:-1])
    
    arr = np.insert(arr, 0,1, axis=1)
    return arr

def get_targets(csv_path):
    lab=pd.read_csv(csv_path)
    labels=np.array(lab.iloc[:,-1])
    return labels

def analytical_solution(feature_matrix, targets, C=0.0):
    dimy=(feature_matrix.shape)[1]
    
    weight_matrix=np.linalg.inv(np.transpose(feature_matrix)@feature_matrix+C*np.identity(dimy))@np.transpose(feature_matrix)@targets
    
    return weight_matrix

def get_predictions(feature_matrix, weights):
    
    y_predicted=feature_matrix@weights

    return y_predicted

def mse_loss(feature_matrix, weights, targets):
    cost = (1/(2*(feature_matrix.shape)[0]))*np.transpose((feature_matrix@weights - targets))@(feature_matrix@weights - targets)
    return cost

def l2_regularizer(weights):
    return np.sum(np.square(weights))

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    C=0.00000000000001 # got 207 keeping C=1e-7
    loss=loss + C * l2_regularizer(weights)
    
    return loss

def initialize_weights(n):
    weights = np.ones(n)
    return weights

def sample_random_batch(feature_matrix, targets, batch_size):
    
    for i in np.arange(0, (feature_matrix.shape)[0], batch_size):
        yield feature_matrix[i:i + batch_size], targets[i:i + batch_size]

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    
    gradients=2*(1/(feature_matrix.shape)[0])*(np.dot(np.transpose(feature_matrix),(np.dot(feature_matrix,weights) - targets)))
    
    return gradients 

def update_weights(weights, gradients, lr):
    
    weights= weights-lr*gradients
       
    return weights

def early_stopping(gradients, weights):
    
    if(np.all(abs(gradients-weights))>0.00000001):
        return 0
    else:
        return 1

def do_gradient_descent(train_feature_matrix,train_targets,dev_feature_matrix,dev_targets,lr=1.0,C=0.0,batch_size=128,
                        max_steps=1000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights
    a sample code is as follows -- 
    '''
    lr=0.01
    
    n=(train_feature_matrix.shape)[1]
    weights = initialize_weights(n)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    dev_loss_function=[]
    train_loss_function=[]
    train_loss_array=[]
    steps_array=[]
    dev_loss_function=np.array(dev_loss_function)
    train_loss_function=np.array(train_loss_function)
    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        
        for features,targets in sample_random_batch(train_feature_matrix,train_targets,batch_size):
         
        #compute gradients
        
            features=np.array(features)
            targets=np.array(targets)

            
            #C=0.00000001
            gradients = compute_gradients(features, weights, targets, C=0.00000001)

            #update weights
            weights = update_weights(weights, gradients, lr)
            
            if step%eval_steps == 0:
                dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
                train_loss = mse_loss(train_feature_matrix, weights, train_targets)
                #print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))
                train_loss_array.append(train_loss)
                steps_array.append(step)
                np.append(train_loss_function,train_loss)
                np.append(dev_loss,dev_loss)
        if(early_stopping(gradients,weights)):
            break
        
        '''
        implement early stopping etc. to improve performance.
        '''
    train_loss_array=np.array(train_loss_array)
    steps_array=np.array(steps_array)
    #print("train_loss function = ",train_loss_array.shape,"number of steps= ",steps_array.shape)
    plt.plot(steps_array,train_loss_array)
    return weights,steps_array,train_loss_array

if __name__ == '__main__':
    train_features, train_targets = get_features('train.csv',True,None), get_targets('train.csv')
    dev_features, dev_targets = get_features('dev.csv',False,None), get_targets('dev.csv')
    a_solution = analytical_solution(train_features, train_targets, C=0.0001)# C=1e-8 
    print('evaluating analytical_solution...')
    
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    print('training LR using gradient descent...')

    # doing feature scaling here 

    dev_features=(dev_features-np.mean(dev_features))/np.std(dev_features)
    train_features=(train_features-np.mean(train_features))/np.std(train_features)

    gradient_descent_soln,steps,train_loss_array = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=5000,
                        eval_steps=5)
    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))


#REFERENCES
    
#https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
#https://towardsdatascience.com/simple-linear-regression-explanation-and-implementation-from-scratch-with-python-26325ca5de1a
#https://towardsdatascience.com/analytical-solution-of-linear-regression-a0e870b038d5
#https://towardsdatascience.com/vectorization-implementation-in-machine-learning-ca652920c55d
#https://www.inf.ed.ac.uk/teaching/courses/mlpr/2016/notes/w1b_linear_regression.pdf
#https://towardsdatascience.com/implementing-sgd-from-scratch-d425db18a72c
#https://medium.com/analytics-vidhya/gradient-descent-intro-and-implementation-in-python-8b6ab0557b7c
#https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/


