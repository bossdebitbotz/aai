#!/usr/bin/env python3
"""
Validate Model Recreation Script

Tests if we can recreate the exact same validation loss as training.
This isolates whether the issue is model recreation vs. data issues.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import json
import math
from torch.utils.data import Dataset, DataLoader

FINAL_DATA_DIR = "data/final_attention"
MODEL_PATH = "models/working_scaled_multigpu/best_model_scaled.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValidationDataset(Dataset):
    """Simple validation dataset matching training."""
    def __init__(self, file_path, target_feature_indices):
        with np.load(file_path, allow_pickle=True) as data:
            self.context = torch.from_numpy(data['contexts']).float()
            self.target_full = torch.from_numpy(data['targets']).float()
            self.target = self.target_full[:, :, target_feature_indices]
            self.len = self.context.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]

class ScaledCrossMarketEmbedding(nn.Module):
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.metadata = embedding_metadata
        self.embed_dim = embed_dim
        
        self.price_embed = nn.Embedding(1, embed_dim // 4)
        self.size_embed = nn.Embedding(1, embed_dim // 4)
        self.exchange_embed = nn.Embedding(4, embed_dim // 4)
        self.pair_embed = nn.Embedding(5, embed_dim // 4)

    def forward(self, num_features):
        embeddings = []
        device = self.price_embed.weight.device
        
        for i in range(num_features):
            feature_embed = torch.cat([
                self.price_embed(torch.tensor(0, device=device)),
                self.size_embed(torch.tensor(0, device=device)),
                self.exchange_embed(torch.tensor(i % 3, device=device)),
                self.pair_embed(torch.tensor(i % 4, device=device))
            ])
            embeddings.append(feature_embed)
        
        return torch.stack(embeddings)

class ScaledBinancePerpEmbedding(nn.Module):
    def __init__(self, embedding_metadata, target_indices, embed_dim):
        super().__init__()
        self.metadata = embedding_metadata
        self.target_indices = target_indices
        self.embed_dim = embed_dim
        
        self.perp_price_embed = nn.Embedding(1, embed_dim // 2)
        self.perp_size_embed = nn.Embedding(1, embed_dim // 2)

    def forward(self, num_target_features):
        embeddings = []
        device = self.perp_price_embed.weight.device
        
        for i in range(num_target_features):
            feature_embed = torch.cat([
                self.perp_price_embed(torch.tensor(0, device=device)),
                self.perp_size_embed(torch.tensor(0, device=device))
            ])
            embeddings.append(feature_embed)
        
        return torch.stack(embeddings)

class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].clone()
        return self.dropout(x)

class ScaledMultiGPUForecaster(nn.Module):
    def __init__(self, embedding_metadata, target_feature_indices, embed_dim, num_heads, 
                 num_encoder_layers, num_decoder_layers, dropout, target_len, num_target_features):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_input_features = embedding_metadata['num_features']
        self.target_feature_indices = target_feature_indices
        self.num_target_features = num_target_features
        self.target_len = target_len
        self.embed_dim = embed_dim

        self.value_projection = nn.Linear(1, embed_dim)
        self.input_embedding = ScaledCrossMarketEmbedding(embedding_metadata, embed_dim)
        self.output_embedding = ScaledBinancePerpEmbedding(embedding_metadata, target_feature_indices, embed_dim)
        self.positional_encoding = ScaledPositionalEncoding(embed_dim, dropout, max_len=100000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout, 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout, 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.transformer_encoder.enable_nested_tensor = False
        self.transformer_decoder.enable_nested_tensor = False

        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, src, tgt):
        input_feature_embeds = self.input_embedding(self.num_input_features)
        output_feature_embeds = self.output_embedding(self.num_target_features)
        
        src_proj = self.value_projection(src.unsqueeze(-1))
        tgt_proj = self.value_projection(tgt.unsqueeze(-1))

        src_embedded = src_proj + input_feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_proj + output_feature_embeds.unsqueeze(0).unsqueeze(0)

        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        src_flat = src_embedded.reshape(batch_size, context_len * self.num_input_features, self.embed_dim)
        src_pos = self.positional_encoding(src_flat)
        memory = self.transformer_encoder(src_pos)
        
        tgt_flat = tgt_embedded.reshape(batch_size, target_len * self.num_target_features, self.embed_dim)
        tgt_pos = self.positional_encoding(tgt_flat)
        
        combined_target_len = target_len * self.num_target_features
        tgt_mask = self.generate_square_subsequent_mask(combined_target_len).to(src.device)
        
        transformer_out = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)

        output = self.output_layer(transformer_out)
        output = output.squeeze(-1)
        output = output.reshape(batch_size, target_len, self.num_target_features)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def get_target_indices(embedding_metadata):
    """Get WLD binance_perp feature indices."""
    target_indices = []
    
    for i, col_name in enumerate(embedding_metadata['columns']):
        col_info = embedding_metadata['column_mapping'][col_name]
        if (col_info['exchange'] == 'binance_perp' and 
            col_info['trading_pair'] == 'WLD-USDT'):
            target_indices.append(i)
    
    return target_indices

def main():
    """Validate model recreation."""
    print("🧪 VALIDATING MODEL RECREATION")
    print("=" * 50)
    
    # Load metadata
    with open(os.path.join(FINAL_DATA_DIR, 'embedding_metadata.json'), 'r') as f:
        embedding_metadata = json.load(f)
    
    target_feature_indices = get_target_indices(embedding_metadata)
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    saved_val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss'))
    model_config = checkpoint['model_config']
    
    print(f"✅ Loaded checkpoint:")
    print(f"  - Saved validation loss: {saved_val_loss:.6f}")
    print(f"  - Model config: {model_config}")
    print(f"  - Target features: {len(target_feature_indices)}")
    
    # Check data shapes first
    with np.load(os.path.join(FINAL_DATA_DIR, 'validation.npz')) as data:
        contexts = data['contexts']
        targets = data['targets']
        print(f"\n📊 Data shapes:")
        print(f"  - Contexts: {contexts.shape}")
        print(f"  - Targets: {targets.shape}")
        actual_target_steps = targets.shape[1]
    
    # Recreate model with ACTUAL data shapes
    model = ScaledMultiGPUForecaster(
        embedding_metadata=embedding_metadata,
        target_feature_indices=target_feature_indices,
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        dropout=0.1,
        target_len=actual_target_steps,  # Use ACTUAL target steps!
        num_target_features=len(target_feature_indices)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model recreated with target_len={actual_target_steps}")
    
    # Load validation dataset with correct target steps
    val_dataset = ValidationDataset(
        os.path.join(FINAL_DATA_DIR, 'validation.npz'),
        target_feature_indices
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Calculate validation loss
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    
    print(f"\n🔄 Computing validation loss...")
    
    with torch.no_grad():
        for batch_idx, (context, target) in enumerate(val_loader):
            context, target = context.to(DEVICE), target.to(DEVICE)
            
            # Create decoder input (shifted target)
            decoder_input = torch.zeros_like(target)
            decoder_input[:, 1:] = target[:, :-1]
            
            # Get predictions
            predictions = model(context, decoder_input)
            
            # Calculate MSE loss
            loss = mse_loss(predictions, target)
            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")
    
    computed_val_loss = total_loss / total_samples
    
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"  - Saved validation loss:    {saved_val_loss:.6f}")
    print(f"  - Computed validation loss: {computed_val_loss:.6f}")
    print(f"  - Difference:               {abs(saved_val_loss - computed_val_loss):.6f}")
    
    if abs(saved_val_loss - computed_val_loss) < 0.001:
        print(f"  ✅ EXCELLENT! Model recreation is correct")
        print(f"  ✅ The issue is likely in the backtest data/setup, not model")
    elif abs(saved_val_loss - computed_val_loss) < 0.01:
        print(f"  ⚠️  CLOSE! Small difference might be due to:")
        print(f"     - Dropout differences (training vs eval mode)")
        print(f"     - Numerical precision")
    else:
        print(f"  ❌ LARGE DIFFERENCE! Model recreation has issues:")
        print(f"     - Wrong target_steps")
        print(f"     - Architecture mismatch") 
        print(f"     - Data loading differences")

if __name__ == "__main__":
    main() 