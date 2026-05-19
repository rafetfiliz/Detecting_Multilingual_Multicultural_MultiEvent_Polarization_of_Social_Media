import os
import glob
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
CONFIG = {
    "CHALLENGE_DIR": "dataset/subtask2/dev/",
    "MAX_LEN": 128,
    "BATCH_SIZE": 32,
    "LABEL_COLS": ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other'],
    
    # trained models
    "TRAINED_MODELS": {
        "xlm-roberta-base": {
            "model_path": "models/xlm-roberta-base_best.bin",
            "model_name": "xlm-roberta-base",
            "thresholds": [0.35, 0.30, 0.30, 0.25, 0.20]  # Adjust based on your validation
        },
        "distilmbert": {
            "model_path": "models/distilmbert_best.bin",
            "model_name": "distilbert-base-multilingual-cased",
            "thresholds": [0.35, 0.30, 0.30, 0.25, 0.20]  # Adjust based on your validation
        },
        "rembert": {
            "model_path": "models/rembert_best.bin",
            "model_name": "google/rembert",
            "thresholds": [0.35, 0.30, 0.30, 0.25, 0.20]  # Adjust based on your validation
        },
        "mdeberta": {
            "model_path": "models/mdeberta_best.bin",
            "model_name": "microsoft/mdeberta-v3-base",
            "thresholds": [0.35, 0.30, 0.30, 0.25, 0.20]  # Adjust based on your validation
        }
    },
    
    "OUTPUT_DIR": "predictions/"
}

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# FUNCTIONS
def load_challenge_data(directory):
    """Load challenge data from directory."""
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not all_files:
        raise ValueError(f"No CSV files found in {directory}!")
    
    print(f"Loading challenge data from {directory}...")
    df_list = []
    
    for filename in all_files:
        lang_code = os.path.basename(filename).split('.')[0]
        df = pd.read_csv(filename)
        if 'text' not in df.columns:
            print(f"Warning: Skipping {filename} - no 'text' column")
            continue
        df['language'] = lang_code
        df_list.append(df)
        print(f"  Loaded {filename}: {len(df)} samples")
    
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"\nTotal challenge samples: {len(combined_df)}")
    print(f"Columns: {combined_df.columns.tolist()}")
    
    return combined_df

class PredictionDataset(Dataset):
    """Simple dataset for predictions."""
    
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten()
        }

def generate_predictions(model_key, model_info, challenge_df, device):
    """Generate predictions for a trained model."""
    
    print(f"Generating predictions: {model_key}")
    
    # Check if model exists
    if not os.path.exists(model_info['model_path']):
        print(f"Model not found: {model_info['model_path']}")
        return None
    
    # Load tokenizer and model
    print(f"Loading model from: {model_info['model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(model_info['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_info['model_name'],
        num_labels=len(CONFIG['LABEL_COLS']),
        problem_type="multi_label_classification"
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_info['model_path'], map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    pred_dataset = PredictionDataset(challenge_df['text'].values, tokenizer, CONFIG['MAX_LEN'])
    pred_loader = DataLoader(pred_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
    
    # Generate predictions
    all_logits = []
    print("Generating predictions...")
    
    with torch.no_grad():
        for d in tqdm(pred_loader):
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits)
    
    # Process predictions
    all_logits = torch.cat(all_logits, dim=0)
    pred_probs = torch.sigmoid(all_logits).cpu().numpy()
    
    # Apply thresholds
    pred_binary = np.zeros_like(pred_probs, dtype=int)
    thresholds = model_info['thresholds']
    
    print(f"\nApplying thresholds: {thresholds}")
    for i, thresh in enumerate(thresholds):
        pred_binary[:, i] = (pred_probs[:, i] > thresh).astype(int)
    
    # Create submission dataframe
    if 'id' in challenge_df.columns:
        submission_df = challenge_df[['id']].copy()
    else:
        print("Warning: 'id' column not found, creating sequential IDs")
        submission_df = pd.DataFrame({'id': range(len(challenge_df))})
    
    for i, label in enumerate(CONFIG['LABEL_COLS']):
        submission_df[label] = pred_binary[:, i]
    
    # Save predictions
    output_file = os.path.join(CONFIG['OUTPUT_DIR'], f"{model_key}_predictions.csv")
    submission_df.to_csv(output_file, index=False)
    
    print(f"\nPredictions saved to: {output_file}")
    print(f"  Total predictions: {len(submission_df)}")
    print(f"\nPrediction distribution:")
    for label in CONFIG['LABEL_COLS']:
        n_positive = submission_df[label].sum()
        pct = n_positive / len(submission_df) * 100
        print(f"  {label:20s}: {n_positive:5d} ({pct:5.1f}%)")
    
    return submission_df

# MAIN
def main():  
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load challenge data
    try:
        challenge_df = load_challenge_data(CONFIG['CHALLENGE_DIR'])
    except Exception as e:
        print(f"\nError loading challenge data: {e}")
        print("\nMake sure:")
        print(f"  1. Directory exists: {CONFIG['CHALLENGE_DIR']}")
        print(f"  2. Contains CSV files with 'text' column")
        return
    
    # Generate predictions for each trained model
    results = {}
    for model_key, model_info in CONFIG['TRAINED_MODELS'].items():
        try:
            submission_df = generate_predictions(model_key, model_info, challenge_df, device)
            if submission_df is not None:
                results[model_key] = submission_df
        except Exception as e:
            print(f"\nError with {model_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("SUMMARY")
    print(f"\nSuccessfully generated predictions for {len(results)} model(s):")
    for model_key in results.keys():
        output_file = os.path.join(CONFIG['OUTPUT_DIR'], f"{model_key}_predictions.csv")
        print(f"{model_key}: {output_file}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()