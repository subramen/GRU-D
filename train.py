# **********************************
# 	Author: Suraj Subramanian
# 	2nd January 2020
# **********************************

import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_utils as du
import utils
import random
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.lines import Line2D
from model import IBDModel
import pandas as pd, numpy as np
import shutil
from sklearn.metrics import classification_report, roc_curve, auc, \
brier_score_loss, precision_recall_curve, average_precision_score, matthews_corrcoef
pd.options.mode.chained_assignment = None

class TrainPlot:
    def __init__(self, modelname):
        self.modelname = modelname
        self.call_count = 0
    
    def plot_grad_flow(self, named_parameters):
        '''
        https://github.com/alwynmathew/gradflow-check
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                try:
                    ave_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())
                except Exception as e:
                    print(f"TRAIN ERROR || param: {n}\n{p}", e)
        
        plt.figure(0)
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(os.path.join('models', self.modelname, 'gradflow', f'{self.call_count}.png'), bbox_inches='tight')

        self.call_count +=1



def save_ckp(state, is_best, checkpoint_path, best_model_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path,  best_model_path)



def train_epoch(model, train_iter, tgt_col, aux_cols, criterion, optimizer, aux_alpha, tr_alpha,\
                scheduler=None, print_every=10, plotter=None):

    def get_aux_loss(pred_aux, true_aux):
        aux_criterion = nn.BCEWithLogitsLoss() # MultiLabel
        combined_aux_loss = 0
        for ix in range(len(true_aux)):
            truth = true_aux[ix].to(device)
            pred = pred_aux[ix]
            if len(truth.size())==1: truth = torch.unsqueeze(truth, 1)
            combined_aux_loss += aux_criterion(pred, truth)
        return combined_aux_loss

    device = utils.try_gpu()
    metrics = utils.Accumulator(5) #batch, loss, outputloss, trloss, auxloss
    batch_size = train_iter.batch_size
    denom = 1+aux_alpha+tr_alpha
    mtl_loss_weights = [1/denom, aux_alpha/denom, tr_alpha/denom]

    for batch, (X, y_dict) in enumerate(train_iter):
        X = X.to(device)  

        # GET LABELS
        true_op = y_dict[tgt_col].to(device)   # OP target tensor  
        true_aux = [y_dict[ac] for ac in aux_cols] # List of Aux target tensors
        
        # GET HIDDENS
        h1 = model.init_hidden(batch_size)
        h2 = model.init_hidden(batch_size)

        # FORWARD PASS
        pred_op, pred_tr, pred_aux, h2 = model(X, h1, h2)   

        # OP LOSS
        op_loss = criterion(pred_op, true_op) # Output Loss
        
        # TR LOSS | Reshape replicated targets and predictions for loss compute. No linear scaling.
        seq_len = X.size(2)
        true_tr = torch.unsqueeze(true_op, 1).repeat(1, seq_len).view(batch_size*seq_len)
        pred_tr = pred_tr.view(batch_size*seq_len, -1) # [batch_size*seq_len, C]
        tr_loss = criterion(pred_tr.to(device), true_tr)

        # AUX LOSS
        aux_loss = get_aux_loss(pred_aux, true_aux)
        
        # COMBINED LOSS
        loss = mtl_loss_weights[0]*op_loss + mtl_loss_weights[1]*aux_loss + mtl_loss_weights[2]*tr_loss # Weighted combination of OP, Aux, TR loss
        
        # BACKPROP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
        # STORE METRICS
        metrics.add(1, loss.item(), op_loss.item(),tr_loss.item(),aux_loss.item())
        
        if batch%print_every == 0:
            plotter.plot_grad_flow(model.named_parameters())
            print(f"Minibatch:{batch}    OPLoss:{metrics[2]/metrics[0]}    TRLoss:{metrics[3]/metrics[0]}    AuxLoss:{metrics[4]/metrics[0]}    AggLoss:{metrics[1]/metrics[0]}        Examples seen: {metrics[0]*batch_size}")
        
    return metrics


    
def train_model(train_iter, valid_iter, X_Mean, tgt_col, aux_cols, epochs, modelname, nb_classes, \
                lr=0.001, aux_alpha=0, tr_alpha=0, class_weights=None, l2=None, model=None, print_every=100):
    """
    Train a GRUD model

    :param train_iter: Train DataLoader
    :param valid_iter: Valid DataLoader
    :param X_Mean: Empirical Mean values for each dimension in the input (only important for variables with missing data)
    :param tgt_col: (str) Name of OP target
    :param aux_cols: list(str) of names of Aux targets. 
    :param epochs: Int of epochs to run
    :param modelname: Unique name for this model
    :param nb_classes: Number of OP target classes
    :param aux_alpha: Weight for Aux Loss
    :param tr_alpha: Weight for TR Loss
    :param class_weights (optional): Weights to scale OP Loss (for skewed datasets)
    """
    device = utils.try_gpu()

    # Set directory for model outputs
    try:
        os.makedirs(os.path.join('models',modelname))
        os.makedirs(os.path.join('models',modelname, 'gradflow'))
    except FileExistsError: pass

    # Initialize plotter class for gradflow
    plotter = TrainPlot(modelname)

    # Initialize model and learners
    class_weights = class_weights or [1]*nb_classes 
    l2 = l2 or 0

    for X,y in train_iter: break
    input_dim = X.size(-1)
    aux_dim = [ (y[aux_c].size(-1) if len(y[aux_c].size())>1 else 1) for aux_c in aux_cols] # if-else for targets with single dimennsion. their size(-1) will be batchsize

    model = IBDModel(input_dim, nb_classes, X_Mean, aux_dim).to(device=device, dtype=torch.float)    
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.85)

    # Store training metrics    
    train_meta = {}
    train_meta['train_losses'] = []
    train_meta['valid_losses'] = []
    train_meta['min_valid_loss'] = sys.maxsize
    train_meta['epoch_results'] = []


    for epoch in range(epochs):
        # TRAIN EPOCH
        t0 = time.time()
        metrics = train_epoch(model, train_iter, tgt_col, aux_cols, criterion, optimizer, aux_alpha, tr_alpha, scheduler, print_every=print_every, plotter=plotter)
        if epoch<200:scheduler.step()
        print(f"Epoch trained in {time.time()-t0}")

        # EVALUATE AGAINST VALIDATION SET
        t0 = time.time()
        eval_scores = eval_model(model, valid_iter, tgt_col, nb_classes)
        train_meta['epoch_results'].append(eval_scores)
        print(f"Evaluation done in {time.time()-t0}")

        t0 = time.time()
        # SAVE CHECKPOINT
        if eval_scores['loss'] < train_meta['min_valid_loss'] or epoch % 20 == 0:
            train_meta['min_valid_loss'] = min(eval_scores['loss'], train_meta['min_valid_loss'])
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckp(checkpoint, True, os.path.join('models', modelname, 'checkpoint.pt'), os.path.join('models', modelname, 'best_model.pt'))
            print(f"Checkpoint created")

        # LOG PROGRESS
        print("\n\n================================================================================================================\n")
        print(f"Epoch: {epoch+1}    TrainLoss: {metrics[1]/metrics[0]}    ValidLoss: {eval_scores['loss']}    ValidAcc:{eval_scores['accuracy']}       WallTime: {datetime.datetime.now()}\n")
        print(eval_scores['conf_matrix'])
        print(pd.DataFrame.from_dict(eval_scores['clf_report']))
        print(eval_scores['brier'])
        print(eval_scores['roc'])
        print("\n\n================================================================================================================\n")
        
        # SAVE TRAINING PROGRESS DATA
        train_meta['train_losses'].append(metrics[1]/metrics[0])
        train_meta['valid_losses'].append(eval_scores['loss'])
        utils.pkl_dump(train_meta, os.path.join('models', modelname, 'trainmeta.dict'))
        print(f"post eval dumping took {time.time()-t0}")

        # PLOT LOSSES
        t0 = time.time()
        plt.figure(1)
        plt.plot(train_meta['train_losses'])
        plt.plot(train_meta['valid_losses'])
        plt.xlabel("Minibatch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join('models', modelname, modelname+'_lossPlot.png'), bbox_inches='tight')
        print(f"plotting took {time.time()-t0}")

    return model



def eval_model(model, test_iter, tgt_col, nb_classes):
    """
    Return dict containing:
    - Log Loss
    - Accuracy
    - Precision, Recall, F1
    - Cohen's Kappa 
    - Matthew's Corr Coef
    - OvA AUC ROC
    - Binary Brier Loss (if multiclass, min and max label are considered)
    - PR Curve
    """
    device = utils.try_gpu()
    test_loss = 0
    accuracy = 0
    loss_criterion = nn.CrossEntropyLoss()
    conf_matrix = torch.zeros(nb_classes, nb_classes, device=device)
    model.to(device)

    model.eval() # No dropout needed 
    with torch.no_grad(): # require_grad = False
        for batch, (X,y_dict) in enumerate(test_iter):
            y = y_dict[tgt_col]
            h1 = model.init_hidden(test_iter.batch_size)
            h2 = model.init_hidden(test_iter.batch_size)

            yhat, h2 = model.predict(X.to(device).float(), h1, h2)
            _, labels = torch.max(yhat, 1)

            test_loss += loss_criterion(yhat.to(device),y.to(device)).item()
            accuracy += (labels.to(device).long()==y.to(device).long()).float().mean()

            for t,p in zip(y.view(-1), labels.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

            # y, yhat, [logits for all classes]
            preds = torch.cat((torch.unsqueeze(y.to(device).float(), 1), 
                            torch.unsqueeze(labels.float(), 1),
                            torch.softmax(yhat,1).float()),  1).to('cpu') 

            
    conf_matrix = conf_matrix.detach()
    accuracy = (accuracy/(batch+1)).item()
    exp_accuracy = sum(conf_matrix.sum(0)/conf_matrix.sum() * conf_matrix.sum(1)/conf_matrix.sum()).item()
    kappa = (accuracy-exp_accuracy)/(1-exp_accuracy)

    eval_scores = {'loss':test_loss/(batch+1), 'accuracy':accuracy, 'conf_matrix':conf_matrix.tolist(), 'kappa':kappa}
    eval_scores.update(clf_report(preds))
    model.train()
    return eval_scores
    


def clf_report(preds):
    # Pad predictions. logits to input sequences
    results = pd.DataFrame(np.array(preds))
    nb_labels = len(results.columns)-2
    results.columns = ['y', 'yhat'] +[f'logit{x}' for x in range(nb_labels)]
    pos_label = nb_labels-1

    # Classification Report
    report = classification_report(results.y, results.yhat, labels=[0,1,2], target_names=['Low', 'Mid', 'High'], output_dict=True)

    # MCC
    mcc = matthews_corrcoef(results.y, results.yhat)

    # Multiclass ROC
    y_dum = pd.get_dummies(results.y).values
    y_hats = results.values[:,2:]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_labels):
        fpr[i], tpr[i], _ = roc_curve(y_dum[:, i], y_hats[:, i])
        roc_auc[i] = round(auc(fpr[i], tpr[i]), 4)

    # Pos vs Neg Brier Loss
    bin_results = results[results.y.isin([results.y.min(), results.y.max()])]
    y_hats_bin = bin_results[['logit0', f'logit{pos_label}']].values
    brier = dict()
    brier[0] = brier_score_loss((1-bin_results.y).abs(), y_hats_bin[:,0])
    brier[pos_label] = brier_score_loss(bin_results.y, y_hats_bin[:,1])
    
    # PR Curve for positive class
    prec, rec, _= precision_recall_curve(y_dum[:,pos_label], y_hats[:,pos_label])
    pr_curve_pos = {'auprc':round(average_precision_score(y_dum[:,pos_label], y_hats[:,pos_label]), 4)}
    thresholds = [x/10 for x in range(10)]
    for i in thresholds:
        pr_curve_pos[f'precision@recall={i}'] = prec[rec>i][-1]
        pr_curve_pos[f'recall@precision={i}'] = rec[prec>i][0]
    
    prec, rec, _= precision_recall_curve(y_dum[:,0], y_hats[:,0])
    pr_curve_neg = {'auprc':round(average_precision_score(y_dum[:,0], y_hats[:,0]), 4)}
    thresholds = [x/10 for x in range(10)]
    for i in thresholds:
        pr_curve_neg[f'precision@recall={i}'] = prec[rec>i][-1]
        pr_curve_neg[f'recall@precision={i}'] = rec[prec>i][0]

    return {'clf_report':report, 'roc':roc_auc, 'brier':brier, 'preds':preds, 'pr_curve_pos':pr_curve_pos, 'mcc':mcc, 'pr_curve_neg':pr_curve_neg}