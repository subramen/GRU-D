# **********************************
#   Author: Suraj Subramanian
#   2nd January 2020
# **********************************


import torch
import torch.nn as nn
import torch.nn.functional as F
import data_utils as du
import utils
import random
import matplotlib.pyplot as plt

class GRUD(nn.Module):
    
    def __init__(self, input_dim=47, hidden_dim=300, output_dim=3, use_decay=True, op_activation=None, x_mean=None):
        super(GRUD, self).__init__()
        self.ctx = utils.try_gpu()
        
        # Assign input and hidden dim
        mask_dim = delta_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_decay = use_decay
        self.x_mean = x_mean
        
        # Output layer
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.op_activation = op_activation if op_activation is not None else lambda x:x #do-nothing
        
        # Linear Combinators
        if use_decay:
            self.R_lin = nn.Linear(input_dim + hidden_dim + mask_dim, hidden_dim)
            self.Z_lin = nn.Linear(input_dim + hidden_dim + mask_dim, hidden_dim)
            self.tilde_lin = nn.Linear(input_dim + hidden_dim + mask_dim, hidden_dim)
            self.gamma_x_lin = nn.Linear(delta_dim, delta_dim)
            self.gamma_h_lin = nn.Linear(delta_dim, hidden_dim)
        else:
            self.R_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.Z_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.tilde_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
            
            
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, device=self.ctx)
    

    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        step_size = inputs.size(2)
        outputs = None
        if self.x_mean==None:
            self.xmean = torch.mean(torch.squeeze(inputs[:,0,:,:]), [0,1]) # X_Mean of current batch

        for t in range(step_size):
            hidden = self.step(inputs[:,:,t:t+1,:], hidden)
        outputs = self.op_activation(self.fc1(hidden))
        return outputs, hidden

    
    def step(self, inputs, h): # inputs = (batch_size x type x dim)
        x, obs_mask, delta, x_tm1 = torch.squeeze(inputs[:,0,:,:]), \
                            torch.squeeze(inputs[:,1,:,:]), \
                            torch.squeeze(inputs[:,2,:,:]), \
                            torch.squeeze(inputs[:,3,:,:])
        
        if self.use_decay:
            gamma_x = torch.exp(-torch.max(torch.zeros(delta.size()), self.gamma_x_lin(delta)))
            gamma_h = torch.exp(-torch.max(torch.zeros(h.size()), self.gamma_h_lin(delta)))
            x = (obs_mask * x) + (1-obs_mask)*( (gamma_x*x_tm1) + (1-gamma_x)*self.xmean ) 
            h = torch.squeeze(gamma_h*h)

        gate_in = torch.cat((x,h,obs_mask), axis=-1)
        z = F.sigmoid(self.Z_lin(gate_in))
        r = F.sigmoid(self.R_lin(gate_in))
        tilde_in = torch.cat((x, r*h, obs_mask), axis=-1)
        tilde = F.tanh(self.tilde_lin(tilde_in))
        h = (1-z)*h + z*tilde
        return h

    
    
def train_model(train_iter, epochs=10):
    device = utils.try_gpu()
    model = GRUD().float().to(device)
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_steps = []
    
    for epoch in range(1, epochs+1):
        avg_loss, l = train_epoch(model, train_iter, criterion, optimizer)
        loss_steps+=l
        print(f"\nEPOCH {epoch}, AvgLoss: {avg_loss}")
        
    plt.plot(loss_steps)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    
    return model



def train_epoch(model, train_iter, criterion, optimizer, print_every=10):
    device = utils.try_gpu()
    metrics = utils.Accumulator(2) #nb_examples, loss,  
    loss_steps = []
        
    for batch, (X, y) in enumerate(train_iter): 
        state = model.init_hidden(train_iter.batch_size)
        y_hat, state = model(X.float(), state.to(device).float())
        
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        
        metrics.add(y.size(0), loss.item())
        loss_steps.append(loss.item())
        
        if batch%print_every == 0:
            print(f"Minibatch:{batch}  Loss:{metrics[1]/metrics[0]} Examples seen: {metrics[0]}")
        
    return metrics[1]/metrics[0], loss_steps