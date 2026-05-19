import os
import glob
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "TRAIN_DIR": "dataset/subtask2/train/",
    "CHALLENGE_DIR": "dataset/subtask2/dev/",  # Unlabeled challenge data
    "MAX_LEN": 128,
    "BATCH_SIZE": 32,
    "LABEL_COLS": ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other'],
    
    # Available models
    "MODELS": {
        "mbert": "bert-base-multilingual-cased",
        "xlm-roberta-base": "xlm-roberta-base",
        "mdeberta": "microsoft/mdeberta-v3-base",
        "rembert": "google/rembert",
        "distilmbert": "distilbert-base-multilingual-cased"
    },
    
    "GENERATE_CHALLENGE_PREDICTIONS": True,
}

TRAIN_CONFIG = {
    "EPOCHS": 5,
    "LEARNING_RATE": 2e-5,
    "MAX_GRAD_NORM": 1.0,
    "WARMUP_RATIO": 0.1,
    "SAVE_DIR": "models/",
    "RESULTS_FILE": "model_comparison_results.csv",
    "PREDICTIONS_DIR": "predictions/",
}

# Create directories
os.makedirs(TRAIN_CONFIG['SAVE_DIR'], exist_ok=True)
os.makedirs(TRAIN_CONFIG['PREDICTIONS_DIR'], exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_merge_data(directory, require_labels=True):
    """Load and merge all CSV files from a directory."""
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not all_files:
        raise ValueError(f"No CSV files found in {directory}!")
    
    df_list = []
    print(f"Loading from {directory}...")
    
    for filename in all_files:
        lang_code = os.path.basename(filename).split('.')[0]
        try:
            df = pd.read_csv(filename)
            if 'text' not in df.columns:
                print(f"Warning: 'text' column not found in {filename}, skipping...")
                continue
            df['language'] = lang_code
            df_list.append(df)
            print(f"  {lang_code}: {len(df)} samples")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not df_list:
        raise ValueError(f"No valid CSV files could be loaded from {directory}!")
    
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"Total: {len(combined_df)} samples")
    
    if require_labels:
        for col in CONFIG['LABEL_COLS']:
            if col not in combined_df.columns:
                print(f"Warning: Label column '{col}' not found, creating with zeros")
                combined_df[col] = 0
        
        combined_df[CONFIG['LABEL_COLS']] = combined_df[CONFIG['LABEL_COLS']].fillna(0).astype(int)
    
    return combined_df

# ============================================================================
# DATASET CLASSES
# ============================================================================
class PolarizationDataset(Dataset):
    """Dataset for multi-label polarization classification."""
    
    def __init__(self, dataframe, tokenizer, max_len, label_columns):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = dataframe.reset_index(drop=True)
        self.texts = self.data['text'].values
        self.targets = self.data[label_columns].values

    def __len__(self):
        return len(self.data)

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
            'mask': inputs['attention_mask'].flatten(),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class PredictionDataset(Dataset):
    """Dataset for generating predictions (no labels)."""
    
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

# ============================================================================
# MODEL UTILITIES
# ============================================================================
def get_model(model_name, num_labels):
    """Load a pre-trained model for multi-label classification."""
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return model

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# ============================================================================
# METRICS
# ============================================================================
def calculate_metrics(pred_logits, true_labels, threshold=0.5):
    """Calculate classification metrics."""
    pred_probs = torch.sigmoid(pred_logits).cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()
    
    pred_binary = (pred_probs > threshold).astype(int)
    
    per_label_f1 = []
    for i in range(true_labels.shape[1]):
        f1 = f1_score(true_labels[:, i], pred_binary[:, i], zero_division=0)
        per_label_f1.append(f1)
    
    return {
        "micro_f1": f1_score(true_labels, pred_binary, average='micro', zero_division=0),
        "macro_f1": f1_score(true_labels, pred_binary, average='macro', zero_division=0),
        "accuracy": accuracy_score(true_labels, pred_binary),
        "per_label_f1": per_label_f1,
        "pred_probs": pred_probs,
        "true_labels": true_labels
    }

def find_optimal_thresholds(pred_probs, true_labels, label_names):
    """Find optimal threshold for each label."""
    print("\nFinding optimal thresholds...")
    
    optimal_thresholds = []
    
    for i, label in enumerate(label_names):
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in np.arange(0.05, 0.96, 0.05):
            pred_binary = (pred_probs[:, i] > thresh).astype(int)
            f1 = f1_score(true_labels[:, i], pred_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        optimal_thresholds.append(best_thresh)
        print(f"  {label:20s}: threshold={best_thresh:.2f}, F1={best_f1:.4f}")
    
    return optimal_thresholds

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, data_loader, optimizer, device, scheduler):
    """Train for one epoch."""
    model.train()
    losses = []
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for d in progress_bar:
        input_ids = d["ids"].to(device)
        attention_mask = d["mask"].to(device)
        targets = d["targets"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['MAX_GRAD_NORM'])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        progress_bar.set_postfix(loss=loss.item())

    return np.mean(losses)

def eval_model(model, data_loader, device):
    """Evaluate the model."""
    model.eval()
    losses = []
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            
            losses.append(outputs.loss.item())
            all_logits.append(outputs.logits)
            all_targets.append(targets)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return np.mean(losses), all_logits, all_targets

# ============================================================================
# PREDICTION GENERATION
# ============================================================================
def generate_challenge_predictions(model, tokenizer, challenge_df, device, model_key, optimal_thresholds):
    """Generate predictions for the challenge set."""
    print("\n" + "="*60)
    print("GENERATING CHALLENGE PREDICTIONS")
    print("="*60)
    
    if challenge_df is None or len(challenge_df) == 0:
        print("No challenge data available. Skipping predictions.")
        return None
    
    if 'id' not in challenge_df.columns:
        print("Warning: 'id' column not found in challenge data")
        challenge_df['id'] = [f"sample_{i}" for i in range(len(challenge_df))]
    
    print(f"Generating predictions for {len(challenge_df)} samples...")
    
    # Create prediction dataset using the class defined at module level
    pred_dataset = PredictionDataset(challenge_df['text'].values, tokenizer, CONFIG['MAX_LEN'])
    
    # FIXED: Set num_workers=0 to avoid multiprocessing issues on Windows
    pred_loader = DataLoader(
        pred_dataset, 
        batch_size=CONFIG['BATCH_SIZE'] * 2, 
        shuffle=False, 
        num_workers=0  # Changed from 2 to 0
    )
    
    # Generate predictions
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for d in tqdm(pred_loader, desc="Predicting"):
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits)
    
    all_logits = torch.cat(all_logits, dim=0)
    pred_probs = torch.sigmoid(all_logits).cpu().numpy()
    
    # Apply optimal thresholds
    pred_binary = np.zeros_like(pred_probs, dtype=int)
    for i, thresh in enumerate(optimal_thresholds):
        pred_binary[:, i] = (pred_probs[:, i] > thresh).astype(int)
    
    # Create submission dataframe
    submission_df = challenge_df[['id']].copy()
    for i, label in enumerate(CONFIG['LABEL_COLS']):
        submission_df[label] = pred_binary[:, i]
    
    # Save predictions
    output_file = os.path.join(TRAIN_CONFIG['PREDICTIONS_DIR'], f"{model_key}_predictions.csv")
    submission_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Predictions saved to: {output_file}")
    print(f"  Total predictions: {len(submission_df)}")
    print(f"\nPrediction distribution:")
    for label in CONFIG['LABEL_COLS']:
        n_positive = submission_df[label].sum()
        pct = n_positive / len(submission_df) * 100
        print(f"  {label:20s}: {n_positive:5d} ({pct:5.1f}%)")
    
    return submission_df

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_model(model_key, train_loader, val_loader, test_loader, device, challenge_df=None, tokenizer=None):
    """Train a single model and return results."""
    
    model_name = CONFIG['MODELS'][model_key]
    print(f"\n{'='*60}")
    print(f"Training Model: {model_key} ({model_name})")
    print(f"{'='*60}\n")
    
    # Initialize model
    model = get_model(model_name, len(CONFIG['LABEL_COLS']))
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])
    total_steps = len(train_loader) * TRAIN_CONFIG['EPOCHS']
    warmup_steps = int(total_steps * TRAIN_CONFIG['WARMUP_RATIO'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_micro_f1': [], 'val_macro_f1': []}
    best_f1 = 0
    save_path = os.path.join(TRAIN_CONFIG['SAVE_DIR'], f"{model_key}_best.bin")
    
    for epoch in range(TRAIN_CONFIG['EPOCHS']):
        print(f"\nEpoch {epoch + 1}/{TRAIN_CONFIG['EPOCHS']}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_logits, val_targets = eval_model(model, val_loader, device)
        metrics = calculate_metrics(val_logits, val_targets)
        
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val Micro F1: {metrics['micro_f1']:.4f}")
        print(f"Val Macro F1: {metrics['macro_f1']:.4f}")
        
        for i, label in enumerate(CONFIG['LABEL_COLS']):
            print(f"  {label:20s}: {metrics['per_label_f1'][i]:.4f}")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_micro_f1'].append(metrics['micro_f1'])
        history['val_macro_f1'].append(metrics['macro_f1'])
        
        # Save best model
        if metrics['micro_f1'] > best_f1:
            print(f"✓ Validation F1 improved from {best_f1:.4f} to {metrics['micro_f1']:.4f}. Saving model...")
            torch.save(model.state_dict(), save_path)
            best_f1 = metrics['micro_f1']
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print(f"FINAL TEST EVALUATION - {model_key}")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load(save_path))
    test_loss, test_logits, test_targets = eval_model(model, test_loader, device)
    
    # Evaluate with default threshold
    print("\n--- With default threshold (0.5) ---")
    final_metrics = calculate_metrics(test_logits, test_targets, threshold=0.5)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Micro F1: {final_metrics['micro_f1']:.4f}")
    print(f"Test Macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"Test Accuracy: {final_metrics['accuracy']:.4f}")
    
    for i, label in enumerate(CONFIG['LABEL_COLS']):
        print(f"  {label:20s}: {final_metrics['per_label_f1'][i]:.4f}")
    
    # Find optimal thresholds on validation set
    print("\n--- Finding optimal thresholds on validation set ---")
    _, val_logits, val_targets = eval_model(model, val_loader, device)
    val_metrics = calculate_metrics(val_logits, val_targets, threshold=0.5)
    
    optimal_thresholds = find_optimal_thresholds(
        val_metrics['pred_probs'], 
        val_metrics['true_labels'], 
        CONFIG['LABEL_COLS']
    )
    
    # Re-evaluate test set with optimal thresholds
    print("\n--- With optimized thresholds ---")
    pred_probs = final_metrics['pred_probs']
    true_labels = final_metrics['true_labels']
    
    pred_binary_optimized = np.zeros_like(pred_probs)
    for i, thresh in enumerate(optimal_thresholds):
        pred_binary_optimized[:, i] = (pred_probs[:, i] > thresh).astype(int)
    
    optimized_micro_f1 = f1_score(true_labels, pred_binary_optimized, average='micro', zero_division=0)
    optimized_macro_f1 = f1_score(true_labels, pred_binary_optimized, average='macro', zero_division=0)
    
    print(f"Test Micro F1 (optimized): {optimized_micro_f1:.4f}")
    print(f"Test Macro F1 (optimized): {optimized_macro_f1:.4f}")
    
    for i, label in enumerate(CONFIG['LABEL_COLS']):
        f1 = f1_score(true_labels[:, i], pred_binary_optimized[:, i], zero_division=0)
        print(f"  {label:20s}: {f1:.4f}")
    
    # Generate challenge predictions
    if CONFIG['GENERATE_CHALLENGE_PREDICTIONS'] and challenge_df is not None and tokenizer is not None:
        submission_df = generate_challenge_predictions(model, tokenizer, challenge_df, device, model_key, optimal_thresholds)
    
    return {
        'model_key': model_key,
        'model_name': model_name,
        'best_val_micro_f1': best_f1,
        'best_val_macro_f1': max(history['val_macro_f1']),
        'test_micro_f1': final_metrics['micro_f1'],
        'test_macro_f1': final_metrics['macro_f1'],
        'test_accuracy': final_metrics['accuracy'],
        'test_micro_f1_optimized': optimized_micro_f1,
        'test_macro_f1_optimized': optimized_macro_f1,
        'optimal_thresholds': optimal_thresholds,
        'history': history
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    
    print("="*60)
    print("POLARIZATION DETECTION - MULTI-MODEL TRAINING")
    print("="*60)
    
    # 1. Load data
    print("\n1. Loading data...")
    try:
        full_train_df = load_and_merge_data(CONFIG['TRAIN_DIR'], require_labels=True)
        
        # Try to load challenge data
        challenge_df = None
        if CONFIG['GENERATE_CHALLENGE_PREDICTIONS']:
            print(f"\nLoading challenge data...")
            if os.path.exists(CONFIG['CHALLENGE_DIR']):
                try:
                    challenge_df = load_and_merge_data(CONFIG['CHALLENGE_DIR'], require_labels=False)
                except Exception as e:
                    print(f"Could not load challenge data: {e}")
            else:
                print(f"Challenge directory not found: {CONFIG['CHALLENGE_DIR']}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Split data into train/val/test (70/15/15)
    print("\n2. Splitting data...")
    
    train_df, temp_df = train_test_split(full_train_df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    print(f"Training set:   {len(train_df)} ({len(train_df)/len(full_train_df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} ({len(val_df)/len(full_train_df)*100:.1f}%)")
    print(f"Test set:       {len(test_df)} ({len(test_df)/len(full_train_df)*100:.1f}%)")
    if challenge_df is not None:
        print(f"Challenge set:  {len(challenge_df)} samples (unlabeled)")
    
    # 3. Get device
    device = get_device()
    
    # 4. Train models
    results = []
    
    # Select which models to train
    # models_to_train = list(CONFIG['MODELS'].keys())  # Train all models
    models_to_train = ["mbert"]  # Train only mBERT
    # Or select specific ones: models_to_train = ["xlm-roberta-base", "mdeberta"]
    
    print(f"\n3. Training {len(models_to_train)} model(s): {models_to_train}")
    
    for model_key in models_to_train:
        try:
            # Initialize tokenizer for this model
            tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODELS'][model_key])
            
            # Create datasets
            train_dataset = PolarizationDataset(train_df, tokenizer, CONFIG['MAX_LEN'], CONFIG['LABEL_COLS'])
            val_dataset = PolarizationDataset(val_df, tokenizer, CONFIG['MAX_LEN'], CONFIG['LABEL_COLS'])
            test_dataset = PolarizationDataset(test_df, tokenizer, CONFIG['MAX_LEN'], CONFIG['LABEL_COLS'])
            
            # FIXED: Set num_workers=0 for Windows compatibility
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'] * 2, shuffle=False, num_workers=0, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'] * 2, shuffle=False, num_workers=0, pin_memory=True)
            
            # Train model
            result = train_model(model_key, train_loader, val_loader, test_loader, device, challenge_df, tokenizer)
            results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error training {model_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. Save comparison results
    if results:
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        results_df = pd.DataFrame([{
            'Model': r['model_key'],
            'Model Name': r['model_name'],
            'Best Val Micro F1': r['best_val_micro_f1'],
            'Best Val Macro F1': r['best_val_macro_f1'],
            'Test Micro F1': r['test_micro_f1'],
            'Test Macro F1': r['test_macro_f1'],
            'Test Micro F1 (opt)': r['test_micro_f1_optimized'],
            'Test Macro F1 (opt)': r['test_macro_f1_optimized'],
        } for r in results])
        
        print(results_df.to_string(index=False))
        
        results_df.to_csv(TRAIN_CONFIG['RESULTS_FILE'], index=False)
        print(f"\nResults saved to {TRAIN_CONFIG['RESULTS_FILE']}")
    
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()