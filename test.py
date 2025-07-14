import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Ensure the script can find your other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilmodule.utils import make_parse
from utilmodule.createmode import create_model
from datasets.load_datasets import h5file_Dataset
from utilmodule.core import test, seed_torch

def main(args):
    """
    Main function to run the testing process.
    """
    seed_torch(2021)

    # --- Create Model ---
    basedmodel, ppo, classifymodel, memory, FusionHisF = create_model(args)

    # --- Load Model from Checkpoint ---
    # The path is now a standard argument and can be checked directly
    if not args.ckpt_path or not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{args.ckpt_path}'. Please provide a valid path using --ckpt_path.")

    print(f"--- Loading model from: {args.ckpt_path} ---")
    checkpoint = torch.load(args.ckpt_path)

    basedmodel.load_state_dict(checkpoint['model_state_dict'])
    FusionHisF.load_state_dict(checkpoint['FusionHisF'])
    classifymodel.load_state_dict(checkpoint['fc'])
    ppo.policy.load_state_dict(checkpoint['policy'])

    # Set models to evaluation mode
    basedmodel.eval()
    classifymodel.eval()
    FusionHisF.eval()
    ppo.policy.eval()

    # --- Load Test Data ---
    print(f"--- Loading test data from: {args.test_h5} ---")
    test_dataset = h5file_Dataset(args.csv, args.test_h5, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Run Evaluation ---
    print("--- Starting evaluation on the test set... ---")
    precision, recall, f1, auc, accuracy = test(args, basedmodel, ppo, classifymodel, FusionHisF, memory, test_dataloader)

    # --- Save Results ---
    res_list = [[accuracy, auc, precision, recall, f1]]
    df = pd.DataFrame(res_list, columns=['acc', 'auc', 'precision', 'recall', 'f1'])

    # Save the results in the same directory as the model checkpoint
    results_path = os.path.join(os.path.dirname(args.ckpt_path), 'test_results.csv')
    df.to_csv(results_path, index=False)

    print("\n--- Test Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Results saved to: {results_path}")
    print("--------------------")

if __name__ == "__main__":
    # Now you can just call make_parse() directly, as it knows about --ckpt_path
    args = make_parse()
    main(args)