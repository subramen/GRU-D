import torch 
from torch.utils.data import DataLoader
from data_utils import SeqDataset
import train
import utils
import argparse
import time, os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="model & logfile name")
parser.add_argument("-b", "--batch_size")
parser.add_argument("-e", "--epochs")
parser.add_argument("-T", "--OP_tgt", help="Name of OP target")
parser.add_argument("-A", "--AUX_tgts", help="Comma-separated names of Aux targets")
parser.add_argument('-a', '--aux_alpha', help='Weight [0,1) for Aux Loss')
parser.add_argument('-t', '--tr_alpha', help='Weight [0,1) for TR Loss')
parser.add_argument('-lr', '--learning_rate')
parser.add_argument('-d', '--datadir')
args = parser.parse_args()


print(f"\n\n\nTrain model with args {args}\n\n")	
aux_cols = args.AUX_tgts.split(',')


# Load Datasets
train_ds, valid_ds, test_ds = SeqDataset(args.datadir, 'train'), SeqDataset(args.datadir, 'valid'), SeqDataset(args.datadir, 'test') 
X_Mean = utils.pkl_load(os.path.join('data', args.datadir, 'x_mean.pkl'))

# Balanced sampling
class_probab = [0.9, 0.037, 0.06] 
reciprocal_weights = [class_probab[train_ds[index][1][args.OP_tgt]] for index in range(len(train_ds))]
weights = (1 / torch.Tensor(reciprocal_weights))
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_ds))

train_iter = DataLoader(train_ds, batch_size=int(args.batch_size), drop_last=True, num_workers=2, sampler=sampler)
test_iter = DataLoader(test_ds, batch_size=len(test_ds), drop_last=True, num_workers=2)


t0 = time.time()
model = train.train_model(train_iter, test_iter, X_Mean, args.OP_tgt, aux_cols, int(args.epochs), args.model_name, 3, float(args.learning_rate), float(args.aux_alpha), float(args.tr_alpha), class_weights=None, l2=0)
print(f'Time taken to train: {time.time()-t0}')
print(f"Saving model to {os.path.join('models', args.model_name, 'model.mdl')}")
torch.save(model.state_dict(), os.path.join('models', args.model_name, 'model.mdl'))
