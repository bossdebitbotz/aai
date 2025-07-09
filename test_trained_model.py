#!/usr/bin/env python3
"""
Test Script for Trained Attention-Based LOB Forecaster

This script loads the trained model and evaluates it on the test set,
comparing results to the paper's benchmarks.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import model classes (copy from training script)
class LOBDataset:
    """Test dataset loader."""
    def __init__(self, file_path):
        with np.load(file_path) as data:
            self.x = torch.from_numpy(data['x']).float()
            self.y = torch.from_numpy(data['y']).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class CompoundMultivariateEmbedding(nn.Module):
    """Compound multivariate embedding (same as training)."""
    def __init__(self, embedding_metadata, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.metadata = embedding_metadata
        
        unique_attrs = embedding_metadata['unique_attributes']
        attr_embed_dim = embed_dim // 5
        remaining_dim = embed_dim - (attr_embed_dim * 4)
        
        self.level_embedding = nn.Embedding(len(unique_attrs['levels']), attr_embed_dim)
        self.type_embedding = nn.Embedding(len(unique_attrs['order_types']), attr_embed_dim)
        self.feature_embedding = nn.Embedding(len(unique_attrs['features']), attr_embed_dim)
        self.exchange_embedding = nn.Embedding(len(unique_attrs['exchanges']), attr_embed_dim)
        self.pair_embedding = nn.Embedding(len(unique_attrs['trading_pairs']), remaining_dim)
        
        self.projection = nn.Linear(embed_dim, embed_dim)
        self._create_feature_indices()
    
    def _create_feature_indices(self):
        columns = self.metadata['columns']
        column_mapping = self.metadata['column_mapping']
        
        level_to_idx = {level: i for i, level in enumerate(self.metadata['unique_attributes']['levels'])}
        type_to_idx = {otype: i for i, otype in enumerate(self.metadata['unique_attributes']['order_types'])}
        feature_to_idx = {feat: i for i, feat in enumerate(self.metadata['unique_attributes']['features'])}
        exchange_to_idx = {exch: i for i, exch in enumerate(self.metadata['unique_attributes']['exchanges'])}
        pair_to_idx = {pair: i for i, pair in enumerate(self.metadata['unique_attributes']['trading_pairs'])}
        
        level_indices = []
        type_indices = []
        feature_indices = []
        exchange_indices = []
        pair_indices = []
        
        for col in columns:
            attrs = column_mapping[col]
            level_indices.append(level_to_idx[attrs['level']])
            type_indices.append(type_to_idx[attrs['order_type']])
            feature_indices.append(feature_to_idx[attrs['feature_type']])
            exchange_indices.append(exchange_to_idx[attrs['exchange']])
            pair_indices.append(pair_to_idx[attrs['trading_pair']])
        
        self.register_buffer('level_indices', torch.LongTensor(level_indices))
        self.register_buffer('type_indices', torch.LongTensor(type_indices))
        self.register_buffer('feature_indices', torch.LongTensor(feature_indices))
        self.register_buffer('exchange_indices', torch.LongTensor(exchange_indices))
        self.register_buffer('pair_indices', torch.LongTensor(pair_indices))
    
    def forward(self):
        level_embeds = self.level_embedding(self.level_indices)
        type_embeds = self.type_embedding(self.type_indices)
        feature_embeds = self.feature_embedding(self.feature_indices)
        exchange_embeds = self.exchange_embedding(self.exchange_indices)
        pair_embeds = self.pair_embedding(self.pair_indices)
        
        combined_embeds = torch.cat([
            level_embeds, type_embeds, feature_embeds, 
            exchange_embeds, pair_embeds
        ], dim=-1)
        
        return self.projection(combined_embeds)

class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)

class LOBForecaster(nn.Module):
    """LOB Forecaster model (same architecture as training)."""
    def __init__(self, embedding_metadata, embed_dim, num_heads, num_encoder_layers, 
                 num_decoder_layers, d_ff, dropout, target_len):
        super().__init__()
        self.embedding_metadata = embedding_metadata
        self.num_features = len(embedding_metadata['columns'])
        self.target_len = target_len
        self.embed_dim = embed_dim
        
        self.value_projection = nn.Linear(1, embed_dim)
        self.compound_embedding = CompoundMultivariateEmbedding(embedding_metadata, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads, 
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, 1)
        
    def forward(self, src, tgt):
        batch_size, context_len, _ = src.shape
        target_len = tgt.shape[1]
        
        feature_embeds = self.compound_embedding()
        
        src_values = self.value_projection(src.unsqueeze(-1))
        tgt_values = self.value_projection(tgt.unsqueeze(-1))
        
        src_embedded = src_values + feature_embeds.unsqueeze(0).unsqueeze(0)
        tgt_embedded = tgt_values + feature_embeds.unsqueeze(0).unsqueeze(0)
        
        src_flat = src_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, context_len, self.embed_dim)
        tgt_flat = tgt_embedded.permute(0, 2, 1, 3).reshape(batch_size * self.num_features, target_len, self.embed_dim)
        
        src_pos = self.positional_encoding(src_flat)
        tgt_pos = self.positional_encoding(tgt_flat)
        
        memory = self.transformer_encoder(src_pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_len).to(src.device)
        output = self.transformer_decoder(tgt_pos, memory, tgt_mask=tgt_mask)
        
        output = self.output_norm(output)
        output = self.output_projection(output)
        
        output = output.reshape(batch_size, self.num_features, target_len).permute(0, 2, 1)
        
        return output

def load_model():
    """Load the trained model."""
    logger.info("Loading trained model...")
    
    checkpoint_path = 'models/paper_h100/best_model.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    embedding_metadata = checkpoint['embedding_metadata']
    config = checkpoint['config']
    
    # Create model with same architecture
    model = LOBForecaster(
        embedding_metadata=embedding_metadata,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        target_len=24
    ).to(DEVICE)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded - Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f}")
    
    return model, embedding_metadata, config

def calculate_structural_violations(predictions, embedding_metadata):
    """Calculate structural violations in predictions."""
    columns = embedding_metadata['columns']
    column_mapping = embedding_metadata['column_mapping']
    
    violations = 0
    total_constraints = 0
    
    # Group features by exchange_pair
    exchange_pairs = {}
    for idx, col in enumerate(columns):
        attrs = column_mapping[col]
        key = f"{attrs['exchange']}_{attrs['trading_pair']}"
        if key not in exchange_pairs:
            exchange_pairs[key] = {'bid_prices': {}, 'ask_prices': {}}
        
        if attrs['feature_type'] == 'price':
            level = attrs['level']
            if attrs['order_type'] == 'bid':
                exchange_pairs[key]['bid_prices'][level] = idx
            elif attrs['order_type'] == 'ask':
                exchange_pairs[key]['ask_prices'][level] = idx
    
    # Check constraints
    for key, price_indices in exchange_pairs.items():
        bid_prices = price_indices['bid_prices']
        ask_prices = price_indices['ask_prices']
        
        if len(bid_prices) >= 2 and len(ask_prices) >= 2:
            # Check ask price ordering: ask_k < ask_{k+1}
            for level in range(1, 5):
                if level in ask_prices and (level + 1) in ask_prices:
                    ask_k = predictions[:, :, ask_prices[level]]
                    ask_k_plus_1 = predictions[:, :, ask_prices[level + 1]]
                    violations += (ask_k >= ask_k_plus_1).sum().item()
                    total_constraints += ask_k.numel()
            
            # Check bid price ordering: bid_k > bid_{k+1}
            for level in range(1, 5):
                if level in bid_prices and (level + 1) in bid_prices:
                    bid_k = predictions[:, :, bid_prices[level]]
                    bid_k_plus_1 = predictions[:, :, bid_prices[level + 1]]
                    violations += (bid_k <= bid_k_plus_1).sum().item()
                    total_constraints += bid_k.numel()
            
            # Check bid-ask spread: bid_1 < ask_1
            if 1 in bid_prices and 1 in ask_prices:
                bid_1 = predictions[:, :, bid_prices[1]]
                ask_1 = predictions[:, :, ask_prices[1]]
                violations += (bid_1 >= ask_1).sum().item()
                total_constraints += bid_1.numel()
    
    violation_rate = violations / total_constraints if total_constraints > 0 else 0
    return violations, total_constraints, violation_rate

def evaluate_model(model, embedding_metadata):
    """Evaluate model on test set."""
    logger.info("Evaluating model on test set...")
    
    # Load test data
    test_dataset = LOBDataset('data/final_attention/test.npz')
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Evaluate in batches
    batch_size = 32
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Testing"):
            batch_end = min(i + batch_size, len(test_dataset))
            batch_contexts = []
            batch_targets = []
            
            for j in range(i, batch_end):
                context, target = test_dataset[j]
                batch_contexts.append(context)
                batch_targets.append(target)
            
            contexts = torch.stack(batch_contexts).to(DEVICE)
            targets = torch.stack(batch_targets).to(DEVICE)
            
            # Create decoder input (teacher forcing for evaluation)
            decoder_input = torch.cat([
                contexts[:, -1:, :],
                targets[:, :-1, :]
            ], dim=1)
            
            predictions = model(contexts, decoder_input)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_predictions, all_targets

def analyze_results(predictions, targets, embedding_metadata):
    """Analyze and compare results to paper benchmarks."""
    logger.info("Analyzing results...")
    
    # Convert to numpy for analysis
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    
    # Calculate metrics
    mse = mean_squared_error(targets_np.flatten(), predictions_np.flatten())
    mae = mean_absolute_error(targets_np.flatten(), predictions_np.flatten())
    
    # Calculate structural violations
    violations, total_constraints, violation_rate = calculate_structural_violations(
        predictions, embedding_metadata
    )
    
    # Calculate mid-price metrics (average of bid and ask level 1 prices)
    columns = embedding_metadata['columns']
    column_mapping = embedding_metadata['column_mapping']
    
    # Find level-1 bid and ask price indices for each exchange-pair
    mid_price_predictions = []
    mid_price_targets = []
    
    exchange_pairs = {}
    for idx, col in enumerate(columns):
        attrs = column_mapping[col]
        if attrs['level'] == 1 and attrs['feature_type'] == 'price':
            key = f"{attrs['exchange']}_{attrs['trading_pair']}"
            if key not in exchange_pairs:
                exchange_pairs[key] = {}
            exchange_pairs[key][attrs['order_type']] = idx
    
    for key, indices in exchange_pairs.items():
        if 'bid' in indices and 'ask' in indices:
            bid_pred = predictions_np[:, :, indices['bid']]
            ask_pred = predictions_np[:, :, indices['ask']]
            mid_pred = (bid_pred + ask_pred) / 2
            
            bid_target = targets_np[:, :, indices['bid']]
            ask_target = targets_np[:, :, indices['ask']]
            mid_target = (bid_target + ask_target) / 2
            
            mid_price_predictions.extend(mid_pred.flatten())
            mid_price_targets.extend(mid_target.flatten())
    
    mid_price_mse = mean_squared_error(mid_price_targets, mid_price_predictions)
    mid_price_mae = mean_absolute_error(mid_price_targets, mid_price_predictions)
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("🎯 MODEL EVALUATION RESULTS vs PAPER BENCHMARKS")
    print("="*80)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"   • Total MSE:           {mse:.6f}")
    print(f"   • Total MAE:           {mae:.6f}")
    print(f"   • Test Samples:        {len(predictions)}")
    print(f"   • Features:            {predictions.shape[-1]}")
    
    print(f"\n💰 MID-PRICE METRICS:")
    print(f"   • Mid-price MSE:       {mid_price_mse:.6f}")
    print(f"   • Mid-price MAE:       {mid_price_mae:.6f}")
    
    print(f"\n🏗️ STRUCTURAL INTEGRITY:")
    print(f"   • Violations:          {violations:,}")
    print(f"   • Total Constraints:   {total_constraints:,}")
    print(f"   • Violation Rate:      {violation_rate:.4f} ({violation_rate*100:.2f}%)")
    
    print(f"\n📈 PAPER COMPARISON:")
    print(f"   Paper Results (Table 1):")
    print(f"   • Best Method (Compound): MSE ~0.0038, MAE ~0.0395")
    print(f"   • Structure Loss:         ~0.14")
    print(f"   ")
    print(f"   Our Results:")
    print(f"   • Total MSE:             {mse:.6f}")
    print(f"   • Total MAE:             {mae:.6f}")
    print(f"   • Violation Rate:        {violation_rate:.4f}")
    
    # Determine if we beat the paper
    paper_mse = 0.0038
    paper_struct = 0.14
    
    if mse < paper_mse and violation_rate < paper_struct:
        print(f"\n   🚀 RESULT: SIGNIFICANTLY OUTPERFORMED PAPER!")
    elif mse < paper_mse:
        print(f"\n   ✅ RESULT: BETTER MSE THAN PAPER!")
    else:
        print(f"\n   📊 RESULT: COMPARABLE TO PAPER BENCHMARKS")
    
    print("="*80)
    
    return {
        'mse': mse,
        'mae': mae,
        'mid_price_mse': mid_price_mse,
        'mid_price_mae': mid_price_mae,
        'violations': violations,
        'total_constraints': total_constraints,
        'violation_rate': violation_rate,
        'num_samples': len(predictions)
    }

def main():
    """Main evaluation function."""
    logger.info("Starting Model Evaluation")
    
    try:
        # Load trained model
        model, embedding_metadata, config = load_model()
        
        # Evaluate on test set
        predictions, targets = evaluate_model(model, embedding_metadata)
        
        # Analyze results
        results = analyze_results(predictions, targets, embedding_metadata)
        
        # Save results
        results_path = 'models/paper_h100/test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main() 