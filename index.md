# 3. Methodology

## 3.1 Problem Formulation

The GreenEarthNet challenge addresses spatio-temporal vegetation forecasting, specifically predicting future Normalized Difference Vegetation Index (NDVI) values from satellite imagery conditioned on past observations and meteorological data. Following the mathematical formulation established in recent literature, we define the task as:

$$\hat{V}^{T+1:T+K} = f(X^{1:T}, C^{1:T+K}, E; \theta)$$

where:
- $V_t \in \mathbb{R}^{H \times W}$ represents the NDVI vegetation state at time $t$
- $X_t \in \mathbb{R}^{H \times W \times B}$ denotes past satellite imagery with $B$ spectral bands
- $C_t \in \mathbb{R}^M$ represents meteorological variables (temperature, precipitation, radiation, etc.)
- $E \in \mathbb{R}^{H \times W}$ is the static elevation map
- $f(\cdot; \theta)$ is the neural network model with parameters $\theta$
- $T$ is the context length (historical observations)
- $K$ is the prediction horizon (target length)

For GreenEarthNet, the spatial resolution is $H = W = 128$ pixels at 20m resolution, with temporal observations at 5-day intervals. The dataset defines multiple evaluation tracks:
- **IID Track**: 50 days context (10 frames), 100 days target (20 frames)
- **OOD-T/OOD-S Tracks**: Same temporal structure, different spatial/temporal distributions
- **Extreme Track**: 100 days context (20 frames), 200 days target (40 frames)
- **Seasonal Track**: 350 days context (70 frames), 700 days target (140 frames)

## 3.2 Pre-trained Foundation Models for Geospatial Data

### 3.2.1 Prithvi Geospatial Foundation Models

The Prithvi family represents state-of-the-art geospatial foundation models developed through the IBM-NASA collaboration. These models leverage Vision Transformer (ViT) architectures with Masked Autoencoder (MAE) pre-training strategies specifically adapted for multi-spectral satellite imagery.

**Prithvi-EO-1.0-100M** is a temporal vision transformer pre-trained on the contiguous United States Harmonized Landsat-Sentinel (HLS) dataset. The model employs self-supervised learning through masked reconstruction, where:
- Input data format: $(B, C, T, H, W)$ supporting temporal sequences
- Pre-training on 6 spectral bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
- Spatial and temporal attention mechanisms
- MAE loss function with masking ratio of 75%

**Prithvi-EO-2.0-600M** represents a significant scaling improvement with 600 million parameters (6× larger than its predecessor). Key enhancements include:
- Training on 4.2M global time series samples from HLS data
- Enhanced temporal and location embeddings
- Multi-scale feature extraction capabilities
- Superior performance on GEO-bench framework (75.6% average score, 8% improvement)
- Demonstrated effectiveness on flood mapping, burn scar detection, and crop classification

**Fine-tuning Strategy for Vegetation Forecasting:**

For the GreenEarthNet task, we propose fine-tuning Prithvi models following these adaptations:

1. **Input Adaptation**: The GreenEarthNet Sentinel-2 bands (B02, B03, B04, B8A) must be mapped to Prithvi's expected input format. We implement band adaptation layers that project the 4-band input to match Prithvi's 6-band pre-training.

2. **Temporal Encoder**: Maintain Prithvi's temporal attention mechanism while adjusting for the 5-daily sampling rate specific to GreenEarthNet.

3. **Decoder Head**: Replace the reconstruction decoder with a forecasting head that predicts future NDVI deviations:
   $$\hat{V}^i = \hat{V}^0 + \delta^i$$
   where $\hat{V}^0$ is the last observed cloud-free NDVI and $\delta^i$ represents predicted deviations.

4. **Meteorological Conditioning**: Integrate E-OBS weather variables through additional embedding layers that fuse with Prithvi's encoded representations.

### 3.2.2 SatMAE: Masked Autoencoder for Satellite Imagery

SatMAE and its enhanced variant SatMAE++ provide alternative pre-training approaches specifically designed for temporal and multi-spectral satellite data. These models address key limitations of standard MAE approaches when applied to Earth observation data.

**Core Innovations:**

1. **Temporal Encoding**: Unlike stacked-temporal approaches, SatMAE incorporates explicit temporal embeddings that allow the model to:
   - Handle irregular sampling intervals
   - Achieve temporal shift invariance
   - Reason about time separations between observations

2. **Multi-Spectral Grouping**: SatMAE groups spectral bands with distinct positional encodings:
   - Group 1: Visible bands (B02, B03, B04)
   - Group 2: NIR bands (B8A and similar)
   - Independent masking across groups enables better reconstruction

3. **SatMAE++ Multi-Scale Enhancement**: The improved version adds:
   - Multi-scale pre-training with convolution-based upsampling
   - Enhanced performance on both optical and multi-spectral imagery
   - 2.5% mAP improvement on BigEarthNet classification

**Implementation for GreenEarthNet:**

```python
# Pseudo-code for SatMAE integration
class SatMAEVegetationForecaster(nn.Module):
    def __init__(self, pretrained_path, forecast_horizon=20):
        self.satmae_encoder = SatMAE.from_pretrained(pretrained_path)
        self.temporal_embedding = TemporalEmbedding(dim=768)
        self.meteo_encoder = MeteorologicalEncoder(input_dim=8, hidden_dim=256)
        self.forecast_head = ForecastingHead(
            embed_dim=768, 
            num_forecasts=forecast_horizon
        )
    
    def forward(self, sat_images, meteo_data, static_features):
        # Encode satellite imagery with temporal information
        sat_features = self.satmae_encoder(sat_images)
        
        # Integrate meteorological data
        meteo_features = self.meteo_encoder(meteo_data)
        
        # Fuse multi-modal features
        fused_features = self.fusion_layer(sat_features, meteo_features)
        
        # Generate forecasts with deviation prediction
        delta_predictions = self.forecast_head(fused_features)
        
        return delta_predictions
```

### 3.2.3 Additional Baseline Architectures

**ConvLSTM and Variants:**

Convolutional Long Short-Term Memory networks combine the spatial feature extraction of CNNs with temporal modeling of LSTMs. The architecture has proven effective for vegetation forecasting:

$$h_t, c_t = \text{ConvLSTM}(x_t, h_{t-1}, c_{t-1})$$

where convolution operations replace matrix multiplications in standard LSTM gates:

$$\begin{aligned}
i_t &= \sigma(W_{xi} * x_t + W_{hi} * h_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} * x_t + W_{hf} * h_{t-1} + b_f) \\
o_t &= \sigma(W_{xo} * x_t + W_{ho} * h_{t-1} + b_o) \\
g_t &= \tanh(W_{xg} * x_t + W_{hg} * h_{t-1} + b_g) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}$$

**SGED-ConvLSTM** (Stochastic Gradient Enhanced Delta-ConvLSTM) extends this with delta prediction frameworks for improved gradient flow.

**Earthformer:**

A space-time transformer specifically designed for Earth system forecasting, employing:
- Cuboid attention mechanisms for efficient spatio-temporal modeling
- Separate spatial and temporal attention blocks
- Multi-scale feature pyramids

**U-Net Architectures:**

Encoder-decoder architectures with skip connections, particularly effective for dense prediction tasks. For vegetation forecasting, temporal U-Net variants stack frames along the channel dimension.

## 3.3 Quantile Regression Neural Networks for Uncertainty Quantification

### 3.3.1 Motivation for Quantile Regression

Traditional neural networks optimize mean squared error, producing point predictions without uncertainty estimates:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$

However, vegetation forecasting requires uncertainty quantification for:
- Decision-making under climate variability
- Identifying prediction confidence in extreme events
- Quantifying aleatoric (data) vs. epistemic (model) uncertainty
- Generating prediction intervals for downstream applications

Quantile regression extends neural networks to predict conditional quantiles of the target distribution, providing comprehensive uncertainty information beyond point estimates.

### 3.3.2 Theoretical Foundation

**Quantile Loss Function:**

For a given quantile $\tau \in (0,1)$, the quantile loss (also called "pinball loss" or "check loss") is defined as:

$\mathcal{L}_\tau(y, \hat{q}_\tau) = \begin{cases}
\tau(y - \hat{q}_\tau) & \text{if } y \geq \hat{q}_\tau \\
(\tau - 1)(y - \hat{q}_\tau) & \text{if } y < \hat{q}_\tau
\end{cases}$

Equivalently: $\mathcal{L}_\tau(y, \hat{q}_\tau) = (\tau - \mathbb{1}_{y < \hat{q}_\tau})(y - \hat{q}_\tau)$

This asymmetric loss penalizes under-prediction and over-prediction differently based on the target quantile. For $\tau = 0.5$ (median), it reduces to the absolute error loss.

**Multi-Quantile Training:**

To obtain a complete predictive distribution, we train the network to simultaneously predict multiple quantiles $\{\tau_1, \tau_2, ..., \tau_Q\}$. Common choices include:
- Lower bounds: $\tau \in \{0.05, 0.10, 0.25\}$
- Median: $\tau = 0.50$
- Upper bounds: $\tau \in \{0.75, 0.90, 0.95\}$

The total loss becomes:

$\mathcal{L}_{\text{total}} = \frac{1}{Q}\sum_{q=1}^Q \mathcal{L}_{\tau_q}(y, \hat{q}_{\tau_q})$

**Prediction Intervals:**

From quantile predictions, we construct $(1-\alpha)$ confidence intervals:

$\text{PI}_{1-\alpha} = [\hat{q}_{\alpha/2}, \hat{q}_{1-\alpha/2}]$

For example, a 90% prediction interval uses $\hat{q}_{0.05}$ and $\hat{q}_{0.95}$.

### 3.3.3 Deep Quantile Regression Architectures

**Architecture Design:**

For spatio-temporal forecasting, we extend the base model to output multiple quantile predictions:

```python
class QuantileRegressionHead(nn.Module):
    def __init__(self, input_dim, spatial_size, forecast_horizon, quantiles):
        super().__init__()
        self.quantiles = quantiles  # e.g., [0.05, 0.25, 0.5, 0.75, 0.95]
        self.num_quantiles = len(quantiles)
        
        # Separate prediction heads for each quantile
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, forecast_horizon, 1)
            )
            for _ in range(self.num_quantiles)
        ])
    
    def forward(self, encoded_features):
        # Shape: (B, input_dim, H, W)
        quantile_predictions = []
        for head in self.quantile_heads:
            # Shape: (B, forecast_horizon, H, W)
            q_pred = head(encoded_features)
            quantile_predictions.append(q_pred)
        
        # Stack along quantile dimension
        # Shape: (B, num_quantiles, forecast_horizon, H, W)
        return torch.stack(quantile_predictions, dim=1)
```

**Non-Crossing Constraints:**

A key challenge in quantile regression is ensuring monotonicity: $\hat{q}_{\tau_1} \leq \hat{q}_{\tau_2}$ for $\tau_1 < \tau_2$. We implement this through:

1. **Soft Constraints** via additional penalty terms:
   $\mathcal{L}_{\text{crossing}} = \sum_{i: \tau_i < \tau_j} \max(0, \hat{q}_{\tau_i} - \hat{q}_{\tau_j})^2$

2. **Parameterization via Increments**:
   $\hat{q}_{\tau_i} = \hat{q}_{\tau_1} + \sum_{j=2}^i \exp(w_j)$
   where $w_j$ are learned parameters and exponential ensures positivity.

### 3.3.4 Huber Quantile Regression

Deep Huber Quantile Regression Networks (DHQRN) generalize standard quantile regression by combining properties of both quantile and expectile regression through the Huber loss:

$\rho_\delta(u) = \begin{cases}
\frac{1}{2}u^2 & \text{if } |u| \leq \delta \\
\delta(|u| - \frac{\delta}{2}) & \text{if } |u| > \delta
\end{cases}$

The Huber quantile loss becomes:

$\mathcal{L}_{\tau,\delta}(y, \hat{q}) = |\tau - \mathbb{1}_{y < \hat{q}}| \cdot \rho_\delta(y - \hat{q})$

This formulation provides robustness to outliers (through the Huber function) while maintaining quantile estimation properties. As $\delta \to 0$, it converges to standard quantile regression; as $\delta \to \infty$, it converges to expectile regression.

### 3.3.5 Conformalized Quantile Regression (CQR)

For improved calibration and guaranteed coverage, we implement Conformalized Quantile Regression:

**Algorithm:**

1. Split training data into proper training set $\mathcal{D}_{\text{train}}$ and calibration set $\mathcal{D}_{\text{cal}}$
2. Train quantile regression model on $\mathcal{D}_{\text{train}}$
3. Compute non-conformity scores on $\mathcal{D}_{\text{cal}}$:
   $s_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(x_i))$
4. Determine calibrated interval width using $(1-\alpha)(1 + 1/|\mathcal{D}_{\text{cal}}|)$ quantile of scores
5. Generate prediction intervals: $[\hat{q}_{\alpha/2}(x) - \hat{s}, \hat{q}_{1-\alpha/2}(x) + \hat{s}]$

CQR provides distribution-free coverage guarantees, ensuring that true values fall within predicted intervals with probability at least $1-\alpha$.

## 3.4 Proposed Hybrid Architecture

### 3.4.1 Overall Framework

We propose a hybrid architecture that combines the representational power of pre-trained foundation models with quantile regression for uncertainty-aware vegetation forecasting:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing                         │
│  • Satellite Images (B02, B03, B04, B8A)                   │
│  • Meteorological Variables (8 features)                    │
│  • Static Elevation (DEM)                                   │
│  • Cloud Mask                                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│           Pre-trained Foundation Encoder                    │
│  Choice of:                                                 │
│  • Prithvi-EO-2.0-600M (Frozen or Fine-tuned)              │
│  • SatMAE++ (Fine-tuned)                                   │
│  • ConvLSTM (Trained from scratch as baseline)             │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│           Fusion Module                                     │
│  • Temporal attention over encoded features                │
│  • Meteorological embedding integration                    │
│  • Static feature injection                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│      Quantile Regression Forecasting Head                  │
│  • Multi-head architecture (one per quantile)              │
│  • Shared backbone with quantile-specific outputs          │
│  • Non-crossing constraints                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                   Output                                    │
│  • Quantile predictions: Q × T × H × W                     │
│  • Point predictions (median): T × H × W                   │
│  • Prediction intervals                                    │
│  • Uncertainty maps                                        │
└─────────────────────────────────────────────────────────────┘
```

### 3.4.2 Training Strategy

**Two-Stage Training:**

*Stage 1: Foundation Model Fine-tuning*
- Initialize with pre-trained weights (Prithvi or SatMAE)
- Fine-tune on GreenEarthNet using standard MSE loss
- Learning rate: 1e-4 with cosine annealing
- Batch size: 16 samples
- Context: 10 frames, Target: 20 frames (IID track)

*Stage 2: Quantile Regression Head Training*
- Freeze or continue fine-tuning encoder at reduced learning rate (1e-5)
- Train quantile heads with multi-quantile loss
- Target quantiles: $\tau \in \{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95\}$
- Add non-crossing penalty with weight $\lambda_{\text{cross}} = 0.01$
- Implement calibration on validation set for CQR

**Loss Function:**

$\mathcal{L}_{\text{total}} = \underbrace{\frac{1}{Q}\sum_{q=1}^Q \mathcal{L}_{\tau_q}}_{\text{Quantile Loss}} + \lambda_{\text{cross}} \underbrace{\mathcal{L}_{\text{crossing}}}_{\text{Non-crossing}} + \lambda_{\text{reg}} \underbrace{\|\theta\|_2^2}_{\text{Regularization}}$

**Optimization:**
- Optimizer: AdamW with weight decay 1e-4
- Gradient clipping: max norm 1.0
- Mixed precision training (FP16) for memory efficiency

### 3.4.3 Handling Cloud Masking

GreenEarthNet includes an improved cloud mask for distinguishing between missing data and clear observations. We integrate this through:

1. **Masked Loss Computation**: Only compute losses on cloud-free pixels
   $\mathcal{L}_{\text{masked}} = \frac{1}{|\mathcal{M}|}\sum_{(t,i,j) \in \mathcal{M}} \mathcal{L}(y_{t,i,j}, \hat{y}_{t,i,j})$
   where $\mathcal{M} = \{(t,i,j) : \text{mask}_{t,i,j} = 0\}$

2. **Learned Masking Tokens**: Replace masked patches with learnable embeddings before encoding

3. **Temporal Interpolation**: For partially clouded sequences, use temporal context to predict missing values

## 3.5 Evaluation Metrics

### 3.5.1 Vegetation Score (Primary Metric)

The GreenEarthNet Vegetation Score is based on normalized Nash-Sutcliffe Efficiency (NSE):

$\text{NSE}(y, \hat{y}) = 1 - \frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t (y_t - \bar{y})^2}$

**Normalization for robust averaging:**
$\text{NNSE} = \frac{1}{2 - \text{NSE}}$

**Aggregation over vegetation pixels:**
$\text{VegScore} = 2 - \frac{1}{\text{mean}(\text{NNSE}_{\text{veg}})}$

where vegetation pixels are defined by land cover classes: Trees, Scrub, or Grassland.
