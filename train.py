import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# ADDED 'calculate_metrics' to this import line
from utilmodule.utils import make_parse, EarlyStopping, save_checkpoint, calculate_metrics
from utilmodule.core import grouping, expand_data, test
from utilmodule.createmode import create_model
from datasets.load_datasets import h5file_Dataset

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    basedmodel, ppo, classifymodel, memory, FusionHisF = create_model(args)
    basedmodel.to(device)
    classifymodel.to(device)
    FusionHisF.to(device)
    ppo.policy.to(device)
    ppo.policy_old.to(device)

    train_dataset = h5file_Dataset(args.csv, args.train_h5, 'train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = h5file_Dataset(args.csv, args.train_h5, 'val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(
        list(basedmodel.parameters()) + list(classifymodel.parameters()) + list(FusionHisF.parameters()),
        lr=args.lr,
        weight_decay=1e-5
    )

    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=25, stop_epoch=100, verbose=True)

    best_val_auc = 0.0
    for epoch in range(args.epoch):
        basedmodel.train()
        classifymodel.train()
        FusionHisF.train()
        ppo.policy.train()
        total_loss = 0.0
        train_labels, train_probs = [], []

        for coords, data, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch} [Train]"):
            coords, data, label = coords.to(device), data.to(device), label.to(device).long()
            if args.type == 'camelyon16':
                data = basedmodel.fc1(data.float())
            else:
                data = data.float()
            
            if args.ape:
                data = data + basedmodel.absolute_pos_embed.expand(1, data.shape[1], -1)

            coords, data, total_T = expand_data(coords, data, action_size=args.action_size, total_steps=args.test_total_T)
            grouping_instance = grouping(action_size=args.action_size)
            optimizer.zero_grad()

            for patch_step in range(total_T):
                restart = patch_step == 0
                action_index_pro, memory = grouping_instance.rlselectindex_grouping(ppo, memory, coords, sigma=0.02, restart=restart)
                features_group, coords, data, memory = grouping_instance.action_make_subbags(ppo, memory, action_index_pro, coords, data, action_size=args.action_size, restart=restart, delete_begin=True)
                results_dict, trandata_ppo, memory, cl_logits = basedmodel(FusionHisF, features_group, memory, coords, mask_ratio=0)
                _ = ppo.select_action(trandata_ppo, memory, restart_batch=restart, training=True)

            W_results_dict, memory = classifymodel(memory)
            W_logits, W_Y_prob = W_results_dict['logits'], W_results_dict['Y_prob']
            
            classification_loss = criterion(W_logits, label)
            
            final_reward = 1.0 if torch.argmax(W_Y_prob, dim=1) == label else -1.0
            num_actions = len(memory.actions)
            rewards_list = [torch.tensor([0.0], dtype=torch.float32, device=device) for _ in range(num_actions - 1)]
            if num_actions > 0:
                rewards_list.append(torch.tensor([final_reward], dtype=torch.float32, device=device))
            memory.rewards = rewards_list
            
            loss = classification_loss
            loss.backward()
            optimizer.step()
            
            if memory.actions:
                ppo.update(memory)
            memory.clear_memory()

            total_loss += loss.item()
            train_labels.append(label.cpu().numpy())
            train_probs.append(W_Y_prob.detach().cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_labels = np.concatenate(train_labels)
        train_probs = np.concatenate(train_probs)
        _, _, _, train_auc, train_accuracy = calculate_metrics(train_labels, train_probs)
        print(f"Epoch {epoch+1}/{args.epoch} - Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}")

        # Validation
        val_precision, val_recall, val_f1, val_auc, val_accuracy = test(args, basedmodel, ppo, classifymodel, FusionHisF, memory, val_loader)
        print(f"Epoch {epoch+1}/{args.epoch} - Val Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            state = {
                'epoch': epoch + 1,
                'model_state_dict': basedmodel.state_dict(),
                'FusionHisF': FusionHisF.state_dict(),
                'fc': classifymodel.state_dict(),
                'policy': ppo.policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc
            }
            save_checkpoint(state, best_acc=val_accuracy, auc=val_auc, checkpoint='./save_model', filename=f'{args.type}_model.pth.tar')

        early_stopping(epoch, -val_auc, basedmodel, classifymodel, FusionHisF, ppo, optimizer, args)
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    args = make_parse()
    
    if not os.path.exists('./save_model'):
        os.makedirs('./save_model')
    main(args)