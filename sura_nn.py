import numpy as np
import torch
from torch import nn
from torch import optim
import math

'''
Read csv and separate the predictors and response
'''
def load_data(file):
    data = np.genfromtxt(file, dtype=np.float64, skip_header = True, delimiter=',')
    X = data[:, 2:]
    y = data[:,1].astype(np.int64)
    return X, y

'''
Build neural network and train it given the training data
'''
def create_network(X, y, hidden_sizes, epochs):
    N = X.shape[0]
    M = X.shape[1]
    # each layer have ReLU activations to prevent divergence
    model = nn.Sequential(nn.Linear(M, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[2], 1),
                      nn.Sigmoid())
    # Define the loss
    criterion = nn.MSELoss()
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    for _ in range(epochs):
        for i in range(N):
            x = torch.Tensor(list(X[i,:]))
            label = torch.Tensor([y[i]])
            optimizer.zero_grad()
            #print(x)
            #print(label)
            output = model(x)
            #print(output)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
    return model

'''
prediction error is given by percentage of mismatching between the actual response and the predicted response
'''
def pred_error(model, X_test, y_true):
    N = X_test.shape[0]
    y = np.zeros(N)
    for i in range(N):
        x = torch.Tensor(list(X[i,:]))
        out = model(x)
        if out >= 0.5:
            y[i] = 1
       
    return np.mean(abs(y-y_true))
    

if __name__ == '__main__':
    #device = torch.device('cpu')
    #if torch.cuda.is_available():
    #    device = torch.device('cuda')
    # X, y = load_data("title.bwords.df.csv")
    X, y = load_data("ab.bwords.df.csv")
    hidden_sizes = [400, 200, 100]
    epochs = 5

    model = create_network(X, y, hidden_sizes, epochs)

    train_err = pred_error(model, X, y)

    print(train_err)

    
    
    


