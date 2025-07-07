#!/usr/bin/env python3
"""
WLD Forecasting Model Training (Many-to-One)

This script defines and trains a model that uses the entire market context
to forecast only the WLD-USDT order books for the next 5 minutes.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
import json
import logging
from tqdm import tqdm
import math

# --- Configuration ---
FINAL_DATA_DIR = "data/final_wld"
MODEL_SAVE_DIR = "models_wld"
BATCH_SIZE = 16 # Adjusted for potentially larger model size
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 5
EMBED_DIM = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("wld_model_training")

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- 1. Data Loading ---

class LOBDataset(Dataset):
    """PyTorch Dataset for loading LOB sequences with lazy loading."""
    def __init__(self, file_path):
        self.file_path = file_path
        with np.load(file_path, allow_pickle=True) as data:
            self.len = data['x'].shape[0]
            self.x_shape = data['x'].shape
            self.y_shape = data['y'].shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with np.load(self.file_path, allow_pickle=True) as data:
            return (torch.from_numpy(data['x'][idx]).float(),
                    torch.from_numpy(data['y'][idx]).float())

# --- 2. Model Components ---

class CompoundMultivariateEmbedding(nn.Module):
    """Embeds features based on their type (exchange, pair, etc.)."""
    def __init__(self, feature_map, embed_dim):
        super().__init__()
        self.feature_map = feature_map
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleDict({
            'exchange': nn.Embedding(len(feature_map['exchange']['map']), embed_dim),
            'trading_pair': nn.Embedding(len(feature_map['trading_pair']['map']), embed_dim),
            'order_type': nn.Embedding(len(feature_map['order_type']['map']), embed_dim),
            'feature_type': nn.Embedding(len(feature_map['feature_type']['map']), embed_dim),
            'level': nn.Embedding(len(feature_map['level']['map']), embed_dim),
        })

    def forward(self, feature_indices):
        indices = torch.LongTensor(feature_indices).to(DEVICE)
        embeds = self.embeddings['exchange'](indices[:, 0]) + \
                 self.embeddings['trading_pair'](indices[:, 1]) + \
                 self.embeddings['order_type'](indices[:, 2]) + \
                 self.embeddings['feature_type'](indices[:, 3]) + \
                 self.embeddings['level'](indices[:, 4])
        return embeds

class PositionalEncoding(nn.Module):
    """Standard positional encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class WLDForecaster(nn.Module):
    """Many-to-one Transformer model for forecasting WLD prices."""
    def __init__(self, feature_map_x, feature_map_y, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout):
        super().__init__()
        self.num_features_y = len(feature_map_y['columns'])

        # Embeddings
        self.value_projection = nn.Linear(1, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.embedding_x = CompoundMultivariateEmbedding(feature_map_x, embed_dim)
        self.embedding_y = CompoundMultivariateEmbedding(feature_map_y, embed_dim)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )

        # Output Layer
        self.output_layer = nn.Linear(embed_dim, self.num_features_y)

    def forward(self, src, tgt):
        # src: (batch, context_len, num_features_x)
        # tgt: (batch, target_len, num_features_y)

        # --- Encoder ---
        feature_embeds_x = self.embedding_x(self.embedding_x.feature_map['feature_indices'])
        src_proj = self.value_projection(src.unsqueeze(-1))
        src_embedded = src_proj + feature_embeds_x
        # Aggregate features by summing their embeddings
        src_agg = src_embedded.sum(dim=2)
        src_agg_pos = self.pos_encoder(src_agg)
        memory = self.transformer.encoder(src_agg_pos)

        # --- Decoder ---
        feature_embeds_y = self.embedding_y(self.embedding_y.feature_map['feature_indices'])
        tgt_proj = self.value_projection(tgt.unsqueeze(-1))
        tgt_embedded = tgt_proj + feature_embeds_y
        # Aggregate target features
        tgt_agg = tgt_embedded.sum(dim=2)
        tgt_agg_pos = self.pos_encoder(tgt_agg)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        
        decoder_output = self.transformer.decoder(tgt_agg_pos, memory, tgt_mask=tgt_mask)

        # --- Output ---
        output = self.output_layer(decoder_output)
        return output

# --- 3. Utilities ---

def create_feature_map(columns):
    """Creates a map of features and their attributes."""
    feature_map = {
        'columns': columns,
        'exchange': {'map': {}, 'rev': []}, 'trading_pair': {'map': {}, 'rev': []},
        'order_type': {'map': {}, 'rev': []}, 'feature_type': {'map': {}, 'rev': []},
        'level': {'map': {}, 'rev': []}, 'feature_indices': []
    }
    def get_or_add(attr, value):
        if value not in feature_map[attr]['map']:
            feature_map[attr]['map'][value] = len(feature_map[attr]['rev'])
            feature_map[attr]['rev'].append(value)
        return feature_map[attr]['map'][value]

    for col_name in columns:
        parts = col_name.split('_')
        feature_map['feature_indices'].append([
            get_or_add('exchange', f"{parts[0]}_{parts[1]}"),
            get_or_add('trading_pair', parts[2]),
            get_or_add('order_type', parts[3]),
            get_or_add('feature_type', parts[4]),
            get_or_add('level', int(parts[5]))
        ])
    return feature_map

def train_epoch(model, dataloader, loss_fn, optimizer, x_cols_all, y_cols_wld):
    model.train()
    total_loss = 0
    wld_indices_in_x = [x_cols_all.index(c) for c in y_cols_wld]

    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Create decoder input: last known WLD values + target sequence shifted
        x_wld_last = x[:, -1:, wld_indices_in_x]
        y_input = torch.cat([x_wld_last, y[:, :-1, :]], dim=1)

        optimizer.zero_grad()
        y_pred = model(x, y_input)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, loss_fn, x_cols_all, y_cols_wld):
    model.eval()
    total_loss = 0
    wld_indices_in_x = [x_cols_all.index(c) for c in y_cols_wld]
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_wld_last = x[:, -1:, wld_indices_in_x]
            y_input = torch.cat([x_wld_last, y[:, :-1, :]], dim=1)
            y_pred = model(x, y_input)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def run_inference(model_path, data_path, scalers_path, x_cols_path, y_cols_path, model_config):
    """Loads the best model and runs prediction on a single test sample."""
    # --- Load Artifacts ---
    logger.info("Loading artifacts for inference...")
    # Load model
    model = WLDForecaster(**model_config).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load data
    test_dataset = LOBDataset(data_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load scalers and column names
    scalers = joblib.load(scalers_path)
    x_cols_all = joblib.load(x_cols_path)
    y_cols_wld = joblib.load(y_cols_path)
    wld_indices_in_x = [x_cols_all.index(c) for c in y_cols_wld]

    # --- Get a Sample and Predict ---
    with torch.no_grad():
        x, y_true = next(iter(test_loader))
        x, y_true = x.to(DEVICE), y_true.to(DEVICE)

        x_wld_last = x[:, -1:, wld_indices_in_x]
        y_input = torch.cat([x_wld_last, y_true[:, :-1, :]], dim=1)
        
        y_pred_scaled = model(x, y_input)

    # --- Inverse Transform and Display ---
    y_true_unscaled = np.zeros_like(y_true.cpu().numpy())
    y_pred_unscaled = np.zeros_like(y_pred_scaled.cpu().numpy())

    for i, col_name in enumerate(y_cols_wld):
        scaler = scalers[col_name]
        y_true_unscaled[0, :, i] = scaler.inverse_transform(y_true.cpu().numpy()[0, :, i].reshape(-1, 1)).flatten()
        y_pred_unscaled[0, :, i] = scaler.inverse_transform(y_pred_scaled.cpu().numpy()[0, :, i].reshape(-1, 1)).flatten()

    logger.info("--- Sample Prediction vs. Ground Truth (Binance Perpetuals) ---")
    
    # Display for the first prediction step (t+1)
    for i, col_name in enumerate(y_cols_wld):
        is_price = 'price' in col_name
        # Prices were transformed with pct_change, so the unscaled value is a percentage change
        unit = '%' if is_price else 'units'
        multiplier = 100 if is_price else 1
        
        true_val = y_true_unscaled[0, 0, i] * multiplier
        pred_val = y_pred_unscaled[0, 0, i] * multiplier
        
        logger.info(f"Feature: {col_name}")
        logger.info(f"  -> True Value (t+1): {true_val:.6f} {unit}")
        logger.info(f"  -> Pred Value (t+1): {pred_val:.6f} {unit}")

# --- 4. Main Execution ---

def main():
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    logger.info("Loading data and creating feature maps...")
    cols_x = joblib.load(os.path.join(FINAL_DATA_DIR, "columns_x.gz"))
    cols_y = joblib.load(os.path.join(FINAL_DATA_DIR, "columns_y.gz"))
    feature_map_x = create_feature_map(cols_x)
    feature_map_y = create_feature_map(cols_y)
    
    with open(os.path.join(MODEL_SAVE_DIR, 'feature_map_x.json'), 'w') as f:
        json.dump(feature_map_x, f, indent=4)
    with open(os.path.join(MODEL_SAVE_DIR, 'feature_map_y.json'), 'w') as f:
        json.dump(feature_map_y, f, indent=4)
        
    train_dataset = LOBDataset(os.path.join(FINAL_DATA_DIR, 'train.npz'))
    val_dataset = LOBDataset(os.path.join(FINAL_DATA_DIR, 'validation.npz'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    logger.info("Initializing model...")
    model = WLDForecaster(
        feature_map_x=feature_map_x, feature_map_y=feature_map_y,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_loss, epochs_no_improve = float('inf'), 0
    history = {'train_loss': [], 'val_loss': []}
    
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        logger.info(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, cols_x, cols_y)
        val_loss = validate_epoch(model, val_loader, loss_fn, cols_x, cols_y)
        scheduler.step(val_loss)

        logger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_wld_model.pth'))
            logger.info("Validation loss improved. Saved new best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info("Early stopping triggered.")
                break
            
    with open(os.path.join(MODEL_SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    logger.info("Training complete.")

    # --- 5. Run Inference on Test Set ---
    logger.info("Running inference on a sample from the test set...")
    run_inference(
        model_path=os.path.join(MODEL_SAVE_DIR, 'best_wld_model.pth'),
        data_path=os.path.join(FINAL_DATA_DIR, 'test.npz'),
        scalers_path=os.path.join(FINAL_DATA_DIR, 'scalers.gz'),
        x_cols_path=os.path.join(FINAL_DATA_DIR, 'columns_x.gz'),
        y_cols_path=os.path.join(FINAL_DATA_DIR, 'columns_y.gz'),
        model_config={
            'feature_map_x': feature_map_x, 'feature_map_y': feature_map_y,
            'embed_dim': EMBED_DIM, 'num_heads': NUM_HEADS,
            'num_encoder_layers': NUM_ENCODER_LAYERS, 'num_decoder_layers': NUM_DECODER_LAYERS,
            'dropout': DROPOUT
        }
    )

if __name__ == "__main__":
    main() 