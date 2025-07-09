#!/usr/bin/env python3
"""
LOB Model Backtesting System

This script backtests the trained attention-based LOB forecaster by:
1. Making sequential predictions on test data
2. Simulating trading decisions based on predictions
3. Calculating trading performance metrics
4. Analyzing prediction accuracy over time
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import model classes (same as test script)
class CompoundMultivariateEmbedding(nn.Module):
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

class LOBBacktester:
    """Comprehensive LOB model backtester."""
    
    def __init__(self, model, embedding_metadata, scaler_path):
        self.model = model
        self.embedding_metadata = embedding_metadata
        self.scaler = self.load_scaler(scaler_path)
        self.results = []
        
    def load_scaler(self, scaler_path):
        """Load the data scaler."""
        import pickle
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    
    def extract_trading_features(self, data, prediction):
        """Extract trading-relevant features from LOB data and predictions."""
        columns = self.embedding_metadata['columns']
        column_mapping = self.embedding_metadata['column_mapping']
        
        # Find level-1 bid/ask prices for each exchange-pair
        exchange_pairs = {}
        for idx, col in enumerate(columns):
            attrs = column_mapping[col]
            if attrs['level'] == 1 and attrs['feature_type'] == 'price':
                key = f"{attrs['exchange']}_{attrs['trading_pair']}"
                if key not in exchange_pairs:
                    exchange_pairs[key] = {}
                exchange_pairs[key][attrs['order_type']] = idx
        
        features = {}
        for key, indices in exchange_pairs.items():
            if 'bid' in indices and 'ask' in indices:
                # Current mid-price
                current_bid = data[-1, indices['bid']]
                current_ask = data[-1, indices['ask']]
                current_mid = (current_bid + current_ask) / 2
                
                # Predicted mid-price (average over prediction horizon)
                pred_bid = prediction[:, indices['bid']].mean()
                pred_ask = prediction[:, indices['ask']].mean()
                pred_mid = (pred_bid + pred_ask) / 2
                
                # Price movement prediction
                price_change = pred_mid - current_mid
                price_change_pct = price_change / current_mid if current_mid != 0 else 0
                
                # Spread analysis
                current_spread = current_ask - current_bid
                pred_spread = pred_ask - pred_bid
                spread_change = pred_spread - current_spread
                
                features[key] = {
                    'current_mid': current_mid,
                    'predicted_mid': pred_mid,
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'current_spread': current_spread,
                    'predicted_spread': pred_spread,
                    'spread_change': spread_change,
                    'current_bid': current_bid,
                    'current_ask': current_ask,
                    'predicted_bid': pred_bid,
                    'predicted_ask': pred_ask
                }
        
        return features
    
    def generate_trading_signals(self, features):
        """Generate trading signals based on predictions."""
        signals = {}
        
        for pair, pair_features in features.items():
            # Signal strength based on predicted price change
            price_change_pct = pair_features['price_change_pct']
            
            # Trading thresholds (configurable)
            strong_signal_threshold = 0.001  # 0.1%
            weak_signal_threshold = 0.0005   # 0.05%
            
            # Generate signal
            if price_change_pct > strong_signal_threshold:
                signal = 'STRONG_BUY'
                confidence = min(abs(price_change_pct) * 1000, 1.0)
            elif price_change_pct > weak_signal_threshold:
                signal = 'BUY'
                confidence = min(abs(price_change_pct) * 500, 1.0)
            elif price_change_pct < -strong_signal_threshold:
                signal = 'STRONG_SELL'
                confidence = min(abs(price_change_pct) * 1000, 1.0)
            elif price_change_pct < -weak_signal_threshold:
                signal = 'SELL'
                confidence = min(abs(price_change_pct) * 500, 1.0)
            else:
                signal = 'HOLD'
                confidence = 0.1
            
            signals[pair] = {
                'signal': signal,
                'confidence': confidence,
                'predicted_return': price_change_pct,
                'spread_favorable': pair_features['spread_change'] < 0  # Narrowing spread is good
            }
        
        return signals
    
    def simulate_trade_execution(self, signals, features, position_size=1000):
        """Simulate trade execution and calculate returns."""
        trades = {}
        
        for pair, signal_data in signals.items():
            if pair not in features:
                continue
                
            pair_features = features[pair]
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            if signal in ['BUY', 'STRONG_BUY']:
                # Buy at ask, sell at predicted bid
                entry_price = pair_features['current_ask']
                exit_price = pair_features['predicted_bid']
                direction = 1
            elif signal in ['SELL', 'STRONG_SELL']:
                # Sell at bid, buy back at predicted ask
                entry_price = pair_features['current_bid']
                exit_price = pair_features['predicted_ask']
                direction = -1
            else:
                continue  # No trade for HOLD
            
            # Calculate trade metrics
            trade_return = (exit_price - entry_price) * direction / entry_price
            trade_pnl = trade_return * position_size * confidence
            
            trades[pair] = {
                'signal': signal,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'trade_return': trade_return,
                'trade_pnl': trade_pnl,
                'confidence': confidence,
                'position_size': position_size * confidence
            }
        
        return trades
    
    def backtest_sequence(self, test_data, batch_size=32):
        """Run backtest on sequential data."""
        logger.info(f"Starting backtest on {len(test_data)} samples...")
        
        all_results = []
        cumulative_pnl = 0
        num_trades = 0
        winning_trades = 0
        
        # Process in smaller batches for memory efficiency
        for i in tqdm(range(0, len(test_data), batch_size), desc="Backtesting"):
            batch_end = min(i + batch_size, len(test_data))
            batch_contexts = []
            batch_targets = []
            
            for j in range(i, batch_end):
                context, target = test_data[j]
                batch_contexts.append(context)
                batch_targets.append(target)
            
            contexts = torch.stack(batch_contexts).to(DEVICE)
            targets = torch.stack(batch_targets).to(DEVICE)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                # For autoregressive prediction (no teacher forcing)
                decoder_input = contexts[:, -1:, :]  # Use last context step as initial decoder input
                predictions = self.model(contexts, decoder_input)
            
            # Process each sample in batch
            for batch_idx in range(len(batch_contexts)):
                context_data = contexts[batch_idx].cpu().numpy()
                target_data = targets[batch_idx].cpu().numpy()
                prediction_data = predictions[batch_idx].cpu().numpy()
                
                # Extract features and generate signals
                features = self.extract_trading_features(context_data, prediction_data)
                signals = self.generate_trading_signals(features)
                trades = self.simulate_trade_execution(signals, features)
                
                # Calculate prediction accuracy
                actual_features = self.extract_trading_features(context_data, target_data)
                
                # Record results
                sample_result = {
                    'sample_idx': i + batch_idx,
                    'features': features,
                    'signals': signals,
                    'trades': trades,
                    'actual_features': actual_features,
                    'timestamp': datetime.now() + timedelta(seconds=(i + batch_idx) * 5)
                }
                
                # Calculate sample metrics
                sample_pnl = sum(trade['trade_pnl'] for trade in trades.values())
                sample_trades = len(trades)
                sample_winning = sum(1 for trade in trades.values() if trade['trade_pnl'] > 0)
                
                cumulative_pnl += sample_pnl
                num_trades += sample_trades
                winning_trades += sample_winning
                
                sample_result['sample_pnl'] = sample_pnl
                sample_result['cumulative_pnl'] = cumulative_pnl
                
                all_results.append(sample_result)
        
        # Calculate overall metrics
        total_return = cumulative_pnl / (len(test_data) * 1000)  # Assuming 1000 base position
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        avg_trade_pnl = cumulative_pnl / num_trades if num_trades > 0 else 0
        
        logger.info(f"Backtest completed:")
        logger.info(f"  Total samples: {len(test_data)}")
        logger.info(f"  Total trades: {num_trades}")
        logger.info(f"  Win rate: {win_rate:.2%}")
        logger.info(f"  Total PnL: ${cumulative_pnl:.2f}")
        logger.info(f"  Total return: {total_return:.2%}")
        logger.info(f"  Avg trade PnL: ${avg_trade_pnl:.2f}")
        
        return all_results, {
            'total_trades': num_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': cumulative_pnl,
            'total_return': total_return,
            'avg_trade_pnl': avg_trade_pnl
        }

def load_model():
    """Load the trained model."""
    logger.info("Loading trained model...")
    
    checkpoint_path = 'models/paper_h100/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    embedding_metadata = checkpoint['embedding_metadata']
    config = checkpoint['config']
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, embedding_metadata, config

def load_test_data():
    """Load test data."""
    logger.info("Loading test data...")
    
    test_data = []
    with np.load('data/final_attention/test.npz') as data:
        contexts = data['x']
        targets = data['y']
        
        for i in range(len(contexts)):
            test_data.append((
                torch.from_numpy(contexts[i]).float(),
                torch.from_numpy(targets[i]).float()
            ))
    
    return test_data

def create_performance_report(results, metrics):
    """Create detailed performance report."""
    print("\n" + "="*100)
    print("🚀 LOB MODEL BACKTESTING RESULTS")
    print("="*100)
    
    print(f"\n📊 TRADING PERFORMANCE:")
    print(f"   • Total Trades:        {metrics['total_trades']:,}")
    print(f"   • Winning Trades:      {metrics['winning_trades']:,}")
    print(f"   • Win Rate:            {metrics['win_rate']:.2%}")
    print(f"   • Total PnL:           ${metrics['total_pnl']:,.2f}")
    print(f"   • Total Return:        {metrics['total_return']:.2%}")
    print(f"   • Avg Trade PnL:       ${metrics['avg_trade_pnl']:,.2f}")
    
    # Analyze by trading pair
    print(f"\n💰 PERFORMANCE BY TRADING PAIR:")
    pair_metrics = {}
    for result in results:
        for pair, trade in result['trades'].items():
            if pair not in pair_metrics:
                pair_metrics[pair] = {'pnl': 0, 'trades': 0, 'wins': 0}
            pair_metrics[pair]['pnl'] += trade['trade_pnl']
            pair_metrics[pair]['trades'] += 1
            if trade['trade_pnl'] > 0:
                pair_metrics[pair]['wins'] += 1
    
    for pair, pm in pair_metrics.items():
        win_rate = pm['wins'] / pm['trades'] if pm['trades'] > 0 else 0
        avg_pnl = pm['pnl'] / pm['trades'] if pm['trades'] > 0 else 0
        print(f"   • {pair:20} | Trades: {pm['trades']:4} | Win Rate: {win_rate:6.2%} | "
              f"Total PnL: ${pm['pnl']:8.2f} | Avg PnL: ${avg_pnl:6.2f}")
    
    # Calculate Sharpe ratio approximation
    daily_returns = []
    current_day_pnl = 0
    samples_per_day = 17280 // 5  # Assuming 24h * 60min * 60sec / 5sec
    
    for i, result in enumerate(results):
        current_day_pnl += result['sample_pnl']
        if (i + 1) % samples_per_day == 0:
            daily_returns.append(current_day_pnl)
            current_day_pnl = 0
    
    if daily_returns:
        avg_daily_return = np.mean(daily_returns)
        daily_volatility = np.std(daily_returns)
        sharpe_ratio = avg_daily_return / daily_volatility if daily_volatility > 0 else 0
        
        print(f"\n📈 RISK METRICS:")
        print(f"   • Avg Daily Return:    ${avg_daily_return:,.2f}")
        print(f"   • Daily Volatility:    ${daily_volatility:,.2f}")
        print(f"   • Sharpe Ratio:        {sharpe_ratio:.3f}")
    
    # Signal accuracy analysis
    correct_predictions = 0
    total_predictions = 0
    
    for result in results:
        for pair in result['signals']:
            if pair in result['actual_features'] and pair in result['features']:
                predicted_direction = result['features'][pair]['price_change']
                actual_direction = result['actual_features'][pair]['price_change']
                
                if (predicted_direction > 0 and actual_direction > 0) or \
                   (predicted_direction < 0 and actual_direction < 0) or \
                   (abs(predicted_direction) < 0.0001 and abs(actual_direction) < 0.0001):
                    correct_predictions += 1
                total_predictions += 1
    
    direction_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"\n🎯 PREDICTION ACCURACY:")
    print(f"   • Direction Accuracy:  {direction_accuracy:.2%}")
    print(f"   • Total Predictions:   {total_predictions:,}")
    print(f"   • Correct Predictions: {correct_predictions:,}")
    
    print("="*100)

def main():
    """Main backtesting function."""
    logger.info("Starting LOB Model Backtesting")
    
    try:
        # Load model and data
        model, embedding_metadata, config = load_model()
        test_data = load_test_data()
        
        # Create backtester
        backtester = LOBBacktester(
            model=model,
            embedding_metadata=embedding_metadata,
            scaler_path='data/final_attention/scaler.pkl'
        )
        
        # Run backtest
        results, metrics = backtester.backtest_sequence(test_data)
        
        # Create performance report
        create_performance_report(results, metrics)
        
        # Save results
        output_dir = 'models/paper_h100'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(f'{output_dir}/backtest_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results (first 100 samples to avoid huge files)
        sample_results = results[:100]
        with open(f'{output_dir}/backtest_sample_results.json', 'w') as f:
            json.dump(sample_results, f, indent=2, default=str)
        
        logger.info(f"Backtest results saved to {output_dir}/")
        
    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        raise

if __name__ == "__main__":
    main() 