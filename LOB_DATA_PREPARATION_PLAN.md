# LOB Data Preparation Plan: Attention-Based Forecasting Model

## 🔍 **Current Status & Investigation Results**

### Database Overview
- **Total Records**: 135,482,962 LOB snapshots
- **Date Range**: May 17, 2025 - July 5, 2025 (49 days)
- **Exchanges**: Binance Spot, Binance Perpetual, Bybit Spot  
- **Trading Pairs**: BTC-USDT, ETH-USDT, SOL-USDT, WLD-USDT
- **Data Structure**: 5-level order book with prices/volumes

### ❌ **Problem Identified: Over-Aggressive Filtering**

**What We Found:**
- Only 6 days out of 49 were marked as having "hour-plus gaps"
- The clean data export only contains 3-4 hours from May 18th
- **48 days have substantial data** (>500k records each)
- Current clean dataset: ~83k records (0.06% of available data)

**Root Cause:**
The clean data filtering was too restrictive, removing entire days instead of handling gaps appropriately.

### 📊 **Available Data Analysis**

**Days with Excellent Coverage (>3M records/day):**
- May 19-31: 13 days
- June 1-16: 16 days  
- June 18-28: 11 days
- **Total: 40 high-quality days**

**Days with Good Coverage (1-3M records/day):**
- May 18: 2.3M records
- June 17, 29-30: 3 days
- July 1-4: 4 days
- **Total: 8 additional days**

**Days with Major Issues (should exclude):**
- May 17: 483k records (partial day)
- July 5: 397k records (partial day)

## 🎯 **Revised Data Preparation Strategy**

### Phase 1: Enhanced Data Extraction
Instead of using overly-filtered clean data, extract from database with smart gap handling:

#### 1.1 Gap Handling Strategy
- **Minor gaps (<5 minutes)**: Forward fill
- **Medium gaps (5-30 minutes)**: Linear interpolation
- **Major gaps (>30 minutes)**: Mark as missing, handle in sequence creation
- **Hour-plus gaps**: Split into separate sessions

#### 1.2 Data Selection Criteria
- **Primary Dataset**: Use 40 high-quality days (>3M records/day)
- **Extended Dataset**: Add 8 good days (1-3M records/day) if needed
- **Minimum Session Length**: 2 hours continuous data
- **Target**: ~130M records across 40-48 days

### Phase 2: Improved Data Processing Pipeline

#### 2.1 Database Extraction
```python
# Extract by date ranges, handling gaps intelligently
# Target: 40-48 days instead of 1 day
# Expected: 100-130M records instead of 83k
```

#### 2.2 Resampling Strategy
- **Frequency**: 5-second intervals (as per paper)
- **Method**: Last value in interval
- **Gap Handling**: Forward fill up to 30 seconds, then mark as missing
- **Session Detection**: Identify continuous periods

#### 2.3 Transformation Pipeline
- **Prices**: Percent-change transformation for stationarity
- **Volumes**: Min-max scaling
- **Missing Data**: Handle gaps without losing entire days
- **Outlier Detection**: Remove extreme values (>3 standard deviations)

### Phase 3: Enhanced Sequence Generation

#### 3.1 Multi-Session Approach
- **Session-Aware Sequences**: Don't cross major gaps
- **Minimum Session Length**: 200 steps (16.7 minutes)
- **Sequence Length**: 120 context + 24 target = 144 steps (12 minutes)
- **Overlap**: 50% overlap between sequences

#### 3.2 Expected Output Scale
- **Raw Data**: ~130M records
- **After Resampling**: ~2.3M 5-second intervals
- **Training Sequences**: ~15,000-20,000 sequences
- **Features**: 160 (8 pairs × 20 features each)

## 🚀 **Implementation Plan**

### Step 1: Create Enhanced Data Extractor
**File**: `extract_full_lob_data.py`
- Query database for 40+ good days
- Smart gap detection and handling
- Session-aware extraction
- Export to parquet by day

### Step 2: Revise Data Preparation Pipeline  
**File**: `prepare_attention_model_data_v2.py`
- Multi-day processing
- Session-aware resampling
- Enhanced gap handling
- Larger sequence generation

### Step 3: Validate Data Quality
- Check feature distributions
- Verify sequence continuity
- Ensure proper scaling
- Test data integrity

### Step 4: Update Model Architecture
- Handle variable-length sessions
- Implement proper masking for gaps
- Scale to larger dataset

## 📈 **Expected Improvements**

### Data Scale Increase
- **Records**: 83k → 130M+ (1,500x increase)
- **Days**: 1 → 40+ (40x increase)  
- **Sequences**: 268 → 15,000+ (55x increase)
- **Training Data**: Substantially more robust

### Model Performance Expected
- **Better Generalization**: More diverse market conditions
- **Improved Stability**: Larger training set
- **Enhanced Forecasting**: Multiple market regimes captured

## 🔧 **Technical Specifications**

### Database Query Strategy
```sql
-- Extract by date with gap analysis
SELECT * FROM lob_snapshots 
WHERE DATE(timestamp) IN ('2025-05-19', '2025-05-20', ...)
AND exchange IN ('binance_spot', 'binance_perp', 'bybit_spot')
ORDER BY timestamp;
```

### Memory Management
- **Batch Processing**: Process 1-2 days at a time
- **Streaming**: Use database cursors for large queries
- **Compression**: Save as compressed parquet files
- **Chunking**: Process in 1M record chunks

### Quality Assurance
- **Continuity Checks**: Verify time series continuity
- **Distribution Analysis**: Check feature distributions
- **Outlier Detection**: Remove extreme values
- **Missing Data Reports**: Document gap locations

## 📅 **Timeline**

### Immediate (Next 2 hours)
1. ✅ **Investigation Complete**: Root cause identified
2. 🚧 **Enhanced Extractor**: Create `extract_full_lob_data.py`
3. 🚧 **Test Extraction**: Validate on 2-3 days first

### Short Term (Next 4 hours)  
4. **Full Extraction**: Process all 40+ good days
5. **Enhanced Pipeline**: Update data preparation
6. **Quality Validation**: Verify data integrity

### Medium Term (Next 8 hours)
7. **Model Architecture**: Update for larger dataset
8. **Training Setup**: Prepare for model training
9. **Baseline Training**: Initial model training

## 🎯 **Success Metrics**

### Data Quality Targets
- **Coverage**: 40+ days of continuous data
- **Completeness**: <5% missing values after gap handling
- **Sequences**: 15,000+ training sequences
- **Features**: 160 properly scaled features

### Model Performance Targets
- **Training Stability**: Smooth convergence
- **Validation Performance**: Consistent improvement
- **Generalization**: Good performance across different market conditions

---

## 🏁 **Next Actions**

1. **Create Enhanced Data Extractor** (`extract_full_lob_data.py`)
2. **Extract 40+ Days of Data** (targeting 100-130M records)
3. **Update Data Preparation Pipeline** (handle larger dataset)
4. **Validate Data Quality** (ensure integrity)
5. **Begin Model Training** (with substantially more data)

**Expected Timeline**: 6-8 hours for complete data preparation
**Expected Dataset Size**: 15,000+ sequences vs current 268 sequences 