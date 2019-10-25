import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.optim as optim

from projected_sinkhorn import conjugate_sinkhorn, projected_sinkhorn
from projected_sinkhorn import wasserstein_cost
from wass_smooth_utils import soft_wass_smooth_forward, wass_predict

def attack(X,y, net, epsilon=0.01, epsilon_iters=10, epsilon_factor=1.1, 
           p=2, kernel_size=5, maxiters=400, 
           alpha=0.1, xmin=0, xmax=1, normalize=lambda x: x, verbose=0, 
           regularization=1000, sinkhorn_maxiters=400, num_samples = 128, num_samples_test = 10000, stdev = .1, batchify_test = 10, alphaprob = .05,
           ball='wasserstein', norm='l2'): 
    batch_size = X.size(0)
    epsilon = X.new_ones(batch_size)*epsilon
    C = wasserstein_cost(X, p=p, kernel_size=kernel_size)
    normalization = X.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
    X_ = X.clone()
    if (num_samples_test is None):
      num_samples_test = num_samples
    X_best = X.clone()
    with torch.no_grad(): 
      if (batchify_test != 0):
        err_best = torch.zeros( X.shape[0], dtype=torch.bool).cuda()
        err = torch.zeros( X.shape[0], dtype=torch.bool).cuda()
        for i in range(0, X.shape[0], batchify_test):
          predictions = wass_predict(X[i:min(i+batchify_test,X.shape[0])],net,stdev,num_samples_test, alphaprob, channel_norm=normalize).cuda()
          incorr = predictions != y[i:min(i+batchify_test,X.shape[0])]
          err_best[i:min(i+batchify_test,X.shape[0])] = err[i:min(i+batchify_test,X.shape[0])] = incorr
      else:
        err = wass_predict(X,net,stdev,num_samples_test, alphaprob, channel_norm=normalize) != y
        err_best = err.clone()
    epsilon_best = epsilon.clone()

    t = 0
    while True: 
        X_noErr = X_[~err]
        X_noErr.requires_grad = True
        y_noErr = y[~err]
        opt = optim.SGD([X_noErr], lr=0.1)
        loss = nn.NLLLoss()(torch.log(soft_wass_smooth_forward(X_noErr, net, num_samples, stdev,  channel_norm = normalize)),y_noErr)
        opt.zero_grad()
        loss.backward()

        with torch.no_grad(): 
            # take a step
            if norm == 'linfinity': 
                X_noErr += alpha*torch.sign(X_noErr.grad)
            elif norm == 'l2': 
                X_noErr += (alpha*X_noErr.grad/(X_noErr.grad.view(X_noErr.size(0),-1).norm(dim=1).view(X_noErr.size(0),1,1,1)))
            elif norm == 'wasserstein': 
                sd_normalization = X_noErr.view(X_noErr.size(0),-1).sum(-1).view(X_noErr.size(0),1,1,1)
                X_noErr = (conjugate_sinkhorn(X_noErr.clone()/sd_normalization, 
                                               X_noErr.grad, C, alpha, regularization, 
                                               verbose=verbose, maxiters=sinkhorn_maxiters
                                               )*sd_normalization)
            else: 
                raise ValueError("Unknown norm")

            # project onto ball
            if ball == 'wasserstein': 
                X_noErr = (projected_sinkhorn(X[~err].clone()/normalization[~err], 
                                          X_noErr.detach()/normalization[~err], 
                                          C,
                                          epsilon[~err],
                                          regularization, 
                                          verbose=verbose, 
                                          maxiters=sinkhorn_maxiters)*normalization[~err])
            elif ball == 'linfinity': 
                X_noErr = torch.min(X_noErr, X[~err] + epsilon[~err].view(X_noErr.size(0), 1, 1,1))
                X_noErr = torch.max(X_noErr, X[~err] - epsilon[~err].view(X_noErr.size(0), 1, 1,1))
            else:
                raise ValueError("Unknown ball")
            X_noErr = torch.clamp(X_noErr, min=xmin, max=xmax)
            if (batchify_test != 0):
              err_noErr = torch.zeros( X_noErr.shape[0], dtype=torch.bool).cuda()
              for i in range(0, X_noErr.shape[0], batchify_test):
                err_noErr[i:min(i+batchify_test,X_noErr.shape[0])] = err_noErr[i:min(i+batchify_test,X_noErr.shape[0])] = wass_predict(X_noErr[i:min(i+batchify_test,X_noErr.shape[0])],net,stdev,num_samples_test, alphaprob, channel_norm=normalize).cuda() != y_noErr[i:min(i+batchify_test,X_noErr.shape[0])]
            else:
              err_noErr = wass_predict(X_noErr,net,stdev,num_samples_test, alphaprob, channel_norm=normalize) != y_noErr
            X_[~err] = X_noErr
            err[~err] = err_noErr
            err_rate = err.sum().item()/batch_size
            if err_rate > err_best.sum().item()/batch_size:
                X_best = X_.clone() 
                err_best = err.clone()
                epsilon_best = epsilon.clone()

            if verbose and t % verbose == 0:
                print(t, loss.item(), epsilon.mean().item(), err_rate)
            
            t += 1
            if err_rate == 1 or t == maxiters: 
                break

            if t > 0 and t % epsilon_iters == 0: 
                epsilon[~err] *= epsilon_factor

    epsilon_best[~err] = float('inf')
    return X_best, err_best, epsilon_best
