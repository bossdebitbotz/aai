#!/usr/bin/env python3
"""
LOB Forecasting Model Training

This script defines and trains the attention-based model for LOB forecasting.
It implements the following key components based on the research paper:
1.  A PyTorch Dataset for loading the pre-processed data.
2.  A Time2Vec embedding layer for temporal encoding.
3.  A Compound Multivariate Embedding layer for spatiotemporal feature encoding.
4.  A Transformer-based Encoder-Decoder architecture.
5.  A combined loss function with a structural regularizer.
6.  A complete training and validation loop with early stopping.
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
FINAL_DATA_DIR = "data/final"
MODEL_SAVE_DIR = "models"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 5  # Early stopping patience
EMBED_DIM = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1
REG_WEIGHT = 0.01 # (w_o) from the paper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_training")

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- 1. Data Loading ---

class LOBDataset(Dataset):
    """PyTorch Dataset for loading LOB sequences with lazy loading."""
    def __init__(self, file_path):
        self.file_path = file_path
        with np.load(file_path) as data:
            self.len = data['x'].shape[0]
            self.x_dtype = data['x'].dtype
            self.y_dtype = data['y'].dtype
            self.x_shape = data['x'].shape
            self.y_shape = data['y'].shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Open the file and read only the requested index
        # This is slower but vastly more memory efficient
        with np.load(self.file_path) as data:
            x = data['x'][idx]
            y = data['y'][idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# --- 2. Model Components ---

class Time2Vec(nn.Module):
    """Time2Vec embedding layer."""
    def __init__(self, in_features, out_features):
        super(Time2Vec, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))

    def forward(self, x):
        v0 = torch.matmul(x, self.w0) + self.b0
        v1 = torch.sin(torch.matmul(x, self.w) + self.b)
        return torch.cat([v0, v1], -1)

class CompoundMultivariateEmbedding(nn.Module):
    """
    Embeds features based on their type (exchange, pair, type, feature, level).
    """
    def __init__(self, feature_map, embed_dim):
        super().__init__()
        self.feature_map = feature_map
        self.embed_dim = embed_dim

        # Create embedding layers for each attribute type
        self.embeddings = nn.ModuleDict({
            'exchange': nn.Embedding(len(feature_map['exchange']['map']), embed_dim),
            'trading_pair': nn.Embedding(len(feature_map['trading_pair']['map']), embed_dim),
            'order_type': nn.Embedding(len(feature_map['order_type']['map']), embed_dim),
            'feature_type': nn.Embedding(len(feature_map['feature_type']['map']), embed_dim),
            'level': nn.Embedding(len(feature_map['level']['map']), embed_dim),
        })

    def forward(self, x_features_indices):
        # x_features_indices: (num_features, num_attributes)
        # Each row corresponds to a feature, columns are its attribute indices
        
        # Get embeddings for each attribute of each feature
        exchange_embeds = self.embeddings['exchange'](x_features_indices[:, 0])
        pair_embeds = self.embeddings['trading_pair'](x_features_indices[:, 1])
        type_embeds = self.embeddings['order_type'](x_features_indices[:, 2])
        feature_embeds = self.embeddings['feature_type'](x_features_indices[:, 3])
        level_embeds = self.embeddings['level'](x_features_indices[:, 4])

        # Combine embeddings by summing them up
        combined_embeds = exchange_embeds + pair_embeds + type_embeds + feature_embeds + level_embeds
        return combined_embeds # Shape: (num_features, embed_dim)

class PositionalEncoding(nn.Module):
    """Standard positional encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LOBForecaster(nn.Module):
    """The main Transformer-based forecasting model."""
    def __init__(self, feature_map, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout, target_len):
        super().__init__()
        self.feature_map = feature_map
        self.num_features = len(feature_map['columns'])
        self.target_len = target_len
        self.embed_dim = embed_dim

        # --- Embedding Layers ---
        self.value_projection = nn.Linear(1, embed_dim)
        self.compound_embedding = CompoundMultivariateEmbedding(feature_map, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        # --- Transformer ---
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        # --- Output Layer ---
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, src, tgt):
        # src shape: (batch_size, context_len, num_features)
        # tgt shape: (batch_size, target_len, num_features)

        # Create feature embeddings (constant for all batches)
        feature_indices = torch.LongTensor(self.feature_map['feature_indices']).to(DEVICE)
        feature_embeds = self.compound_embedding(feature_indices) # (num_features, embed_dim)
        
        # Project values and add feature embeddings
        src_proj = self.value_projection(src.unsqueeze(-1)) # (batch, context_len, num_feat, embed_dim)
        tgt_proj = self.value_projection(tgt.unsqueeze(-1)) # (batch, target_len, num_feat, embed_dim)

        src_embedded = src_proj + feature_embeds
        tgt_embedded = tgt_proj + feature_embeds

        # Reshape for transformer: (batch_size * num_features, seq_len, embed_dim)
        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        src_flat = src_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, context_len, self.embed_dim)
        tgt_flat = tgt_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, target_len, self.embed_dim)
        
        # Add positional encoding
        src_pos = self.positional_encoding(src_flat.permute(1, 0, 2)).permute(1, 0, 2)
        tgt_pos = self.positional_encoding(tgt_flat.permute(1, 0, 2)).permute(1, 0, 2)
        
        # Create target mask for decoder
        tgt_mask = self.transformer.generate_square_subsequent_mask(target_len).to(DEVICE)
        
        # Pass through transformer
        transformer_out = self.transformer(src_pos, tgt_pos, tgt_mask=tgt_mask) # (batch*num_feat, target_len, embed_dim)

        # Project to output
        output = self.output_layer(transformer_out) # (batch*num_feat, target_len, 1)
        
        # Reshape back to original format
        output = output.reshape(batch_size, self.num_features, target_len).permute(0, 2, 1)
        
        return output

# --- 3. Loss Function ---

class StructuralLoss(nn.Module):
    """Calculates the structural regularizer loss."""
    def __init__(self, feature_map):
        super().__init__()
        self.feature_map = feature_map
        self.relu = nn.ReLU()

    def forward(self, y_pred):
        # y_pred shape: (batch, target_len, num_features)
        loss = 0
        
        for ex, pairs in self.feature_map['structure'].items():
            for pair, p_indices in pairs.items():
                # p_indices has keys 'ask_prices', 'bid_prices' with lists of column indices
                
                # Get predicted prices for this instrument
                if p_indices['ask_prices'] and p_indices['bid_prices']:
                    ask_preds = y_pred[:, :, p_indices['ask_prices']]
                    bid_preds = y_pred[:, :, p_indices['bid_prices']]
                    
                    # Rule 1: ask_k < ask_{k+1}  =>  ReLU(ask_k - ask_{k+1})
                    loss += self.relu(ask_preds[:, :, :-1] - ask_preds[:, :, 1:]).mean()

                    # Rule 2: bid_k > bid_{k+1}  =>  ReLU(bid_{k+1} - bid_k)
                    loss += self.relu(bid_preds[:, :, 1:] - bid_preds[:, :, :-1]).mean()

                    # Rule 3: bid_1 < ask_1
                    loss += self.relu(bid_preds[:, :, 0] - ask_preds[:, :, 0]).mean()

        return loss

# --- 4. Training Utilities ---

def create_feature_map(columns):
    """Creates a map of features and their attributes."""
    feature_map = {
        'columns': columns,
        'exchange': {'map': {}, 'rev': []},
        'trading_pair': {'map': {}, 'rev': []},
        'order_type': {'map': {}, 'rev': []},
        'feature_type': {'map': {}, 'rev': []},
        'level': {'map': {}, 'rev': []},
        'feature_indices': [],
        'structure': {}
    }

    def get_or_add(attr, value):
        if value not in feature_map[attr]['map']:
            feature_map[attr]['map'][value] = len(feature_map[attr]['rev'])
            feature_map[attr]['rev'].append(value)
        return feature_map[attr]['map'][value]

    for i, col_name in enumerate(columns):
        # Correctly parse column names like 'binance_spot_BTC-USDT_bid_price_1'
        parts = col_name.split('_')
        exchange = parts[0] + '_' + parts[1]
        trading_pair = parts[2]
        order_type = parts[3]
        feature_type = parts[4]
        level = parts[5]

        # Get indices for each attribute
        ex_idx = get_or_add('exchange', exchange)
        pair_idx = get_or_add('trading_pair', trading_pair)
        type_idx = get_or_add('order_type', order_type)
        feat_idx = get_or_add('feature_type', feature_type)
        level_idx = get_or_add('level', int(level))

        feature_map['feature_indices'].append([ex_idx, pair_idx, type_idx, feat_idx, level_idx])
        
        # Store indices for structural loss
        if feature_type == 'price':
            if exchange not in feature_map['structure']:
                feature_map['structure'][exchange] = {}
            if trading_pair not in feature_map['structure'][exchange]:
                feature_map['structure'][exchange][trading_pair] = {'ask_prices': [], 'bid_prices': []}
            
            if order_type == 'ask':
                feature_map['structure'][exchange][trading_pair]['ask_prices'].append(i)
            else: # bid
                feature_map['structure'][exchange][trading_pair]['bid_prices'].append(i)

    # Sort the price indices by level
    for ex in feature_map['structure']:
        for pair in feature_map['structure'][ex]:
            # Sort by the level, which is the last part of the column name
            feature_map['structure'][ex][pair]['ask_prices'].sort(key=lambda i: int(columns[i].split('_')[-1]))
            feature_map['structure'][ex][pair]['bid_prices'].sort(key=lambda i: int(columns[i].split('_')[-1]))

    return feature_map


def train_epoch(model, dataloader, mse_loss_fn, struct_loss_fn, optimizer, reg_weight):
    """A single training epoch."""
    model.train()
    total_loss, total_mse, total_struct = 0, 0, 0

    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Decoder target needs to be shifted
        y_input = torch.cat([x[:, -1:, :], y[:, :-1, :]], dim=1)

        optimizer.zero_grad()
        
        y_pred = model(x, y_input)
        
        mse = mse_loss_fn(y_pred, y)
        struct = struct_loss_fn(y_pred)
        loss = mse + reg_weight * struct
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_struct += struct.item()

    return total_loss / len(dataloader), total_mse / len(dataloader), total_struct / len(dataloader)


def validate_epoch(model, dataloader, mse_loss_fn, struct_loss_fn, reg_weight):
    """A single validation epoch."""
    model.eval()
    total_loss, total_mse, total_struct = 0, 0, 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_input = torch.cat([x[:, -1:, :], y[:, :-1, :]], dim=1)
            
            y_pred = model(x, y_input)
            
            mse = mse_loss_fn(y_pred, y)
            struct = struct_loss_fn(y_pred)
            loss = mse + reg_weight * struct
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_struct += struct.item()
            
    return total_loss / len(dataloader), total_mse / len(dataloader), total_struct / len(dataloader)


# --- 5. Main Execution ---

def main():
    """Main function to run the training process."""
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # Load data and create feature map
    logger.info("Loading data...")
    columns = joblib.load(os.path.join(FINAL_DATA_DIR, "columns.gz"))
    feature_map = create_feature_map(columns)
    
    # Save feature map
    with open(os.path.join(MODEL_SAVE_DIR, 'feature_map.json'), 'w') as f:
        json.dump(feature_map, f, indent=4)
        
    train_dataset = LOBDataset(os.path.join(FINAL_DATA_DIR, 'train.npz'))
    val_dataset = LOBDataset(os.path.join(FINAL_DATA_DIR, 'validation.npz'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model and loss functions
    logger.info("Initializing model...")
    target_len = val_dataset.y_shape[1]
    
    model = LOBForecaster(
        feature_map=feature_map,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT,
        target_len=target_len
    ).to(DEVICE)
    
    mse_loss_fn = nn.MSELoss()
    struct_loss_fn = StructuralLoss(feature_map).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        logger.info(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss, train_mse, train_struct = train_epoch(model, train_loader, mse_loss_fn, struct_loss_fn, optimizer, REG_WEIGHT)
        val_loss, val_mse, val_struct = validate_epoch(model, val_loader, mse_loss_fn, struct_loss_fn, REG_WEIGHT)
        
        scheduler.step(val_loss)

        logger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        logger.info(f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        logger.info(f"Train Struct Loss: {train_struct:.6f} | Val Struct Loss: {val_struct:.6f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            logger.info("Validation loss improved. Saved new best model.")
        else:
            epochs_no_improve += 1
            logger.info(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve >= PATIENCE:
            logger.info("Early stopping triggered.")
            break
            
    # Save training history
    with open(os.path.join(MODEL_SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f)
        
    logger.info("Training complete.")


if __name__ == "__main__":
    main() 