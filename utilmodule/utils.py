import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import os
import shutil
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='camelyon16', type=str)
    parser.add_argument('--mode', default='rlselect', type=str)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--in_chans', default=1024, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--attn', default='normal', type=str)
    parser.add_argument('--gm', default='cluster', type=str)
    parser.add_argument('--cls', default=True, type=bool)
    parser.add_argument('--num_msg', default=1, type=int)
    parser.add_argument('--ape', default=True, type=bool)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--instaceclass', default=True, type=bool, help='')
    parser.add_argument('--CE_CL', default=True, type=bool, help='')
    parser.add_argument('--ape_class', default=False, type=bool, help='')
    parser.add_argument('--test_h5', default='CAMELYON16/C16-test', type=str)
    parser.add_argument('--train_h5', default='CAMELYON16/C16-train', type=str)
    parser.add_argument('--csv', default='CAMELYON16/camelyon16_test.csv', type=str)
    parser.add_argument('--policy_hidden_dim', type=int, default=512)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--state_dim', type=int, default=512)
    parser.add_argument('--action_size', type=int, default=512)
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.1)
    parser.add_argument('--K_epochs', type=int, default=3)
    parser.add_argument('--test_total_T', type=int, default=1)
    parser.add_argument('--reward_rule', type=str, default="cl", help=' ')
    parser.add_argument('--overfit', action='store_true', help='Flag to indicate an overfitting test run')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the saved model checkpoint for testing')
    args, _ = parser.parse_known_args() # Use parse_known_args to avoid conflicts
    return args

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    schedule_per_epoch = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            value = np.linspace(start_warmup_value, base_value, warmup_epochs)[epoch]
        else:
            iters_passed = epoch * niter_per_ep
            alpha = 0.5 * (1 + np.cos(np.pi * iters_passed / (epochs * niter_per_ep)))
            value = final_value + (base_value - final_value) * alpha
        schedule_per_epoch.append(value)
    return schedule_per_epoch

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error

def calculate_metrics(targets, probs):
    threshold = 0.5
    predictions = (probs[:, 1] >= threshold).astype(int)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    auc = roc_auc_score(targets, probs[:, 1])
    accuracy = accuracy_score(targets, predictions)
    return precision, recall, f1, auc, accuracy

def cat_msg2cluster_group(x_groups,msg_tokens):
    x_groups_cated = []
    for x in x_groups:
        x = x.unsqueeze(dim=0)
        try:
            temp = torch.cat((msg_tokens,x),dim=2)
        except Exception as e:
            print('Error when cat msg tokens to sub-bags')
        x_groups_cated.append(temp)
    return x_groups_cated

def split_array(array, m):
    n = len(array)
    indices = np.random.choice(n, n, replace=False)
    split_indices = np.array_split(indices, m)
    result = []
    for indices_part in split_indices:
        result.append(array[indices_part])
    return result

class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    # MODIFIED: The function now accepts all model parts
    def __call__(self, epoch, val_loss, basedmodel, classifymodel, FusionHisF, ppo, optimizer, args):
        score = -val_loss
        
        # This is the state dictionary that will be saved, matching the main save function
        state = {
            'epoch': epoch + 1,
            'model_state_dict': basedmodel.state_dict(),
            'FusionHisF': FusionHisF.state_dict(),
            'fc': classifymodel.state_dict(),
            'policy': ppo.policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_auc': -score  # val_loss is negative val_auc
        }

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(state, args, epoch)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(state, args, epoch)
            self.counter = 0

    # MODIFIED: This function now saves the complete state dictionary
    def save_checkpoint(self, state, args, epoch):
        if self.verbose:
             print(f'Validation loss improved ({self.val_loss_min:.6f} --> {-self.best_score:.6f}). Saving model...')
        
        # Define a consistent naming scheme
        filename = f"{args.type}_early_stopped_model_epoch_{epoch}.pth.tar"
        save_path = './save_model' # Save to the main model directory
        filepath = os.path.join(save_path, filename)
        
        # Use the main save_checkpoint function to ensure consistency
        save_checkpoint(state, best_acc=0, auc=-self.best_score, checkpoint=save_path, filename=filename)
        self.val_loss_min = -self.best_score

def save_checkpoint(state, best_acc, auc, checkpoint, filename='checkpoint.pth.tar'):
    # Ensure the directory exists
    os.makedirs(checkpoint, exist_ok=True)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    print(f"--- Model Saved to {filepath} ---")