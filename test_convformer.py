import tensorflow as tf
import numpy as np

from convformer import ConvFormer, CombinedLoss

def test_convformer_initialization():
    model = ConvFormer(forecast_horizon=20, quantiles=[0.1, 0.5, 0.9])
    assert model is not None

def test_convformer_forward_pass():
    # Mock data
    batch_size = 2
    input_days = 50
    h, w = 128, 128
    
    sentinel2 = tf.random.normal((batch_size, input_days, h, w, 4))
    weather = tf.random.normal((batch_size, input_days, 8))
    dem = tf.random.normal((batch_size, h, w, 3))
    
    inputs = {
        'sentinel2': sentinel2,
        'weather': weather,
        'dem': dem
    }
    
    model = ConvFormer(forecast_horizon=20, quantiles=[0.1, 0.5, 0.9])
    
    # Run forward pass
    outputs = model(inputs)
    
    # Check output shape
    # Expected: (B, NumQuantiles, ForecastHorizon, H, W, 1)
    # (2, 3, 20, 128, 128, 1)
    expected_shape = (batch_size, 3, 20, h, w, 1)
    
    assert outputs.shape == expected_shape
    print(f"Output shape verified: {outputs.shape}")

def test_combined_loss():
    batch_size = 2
    h, w = 128, 128
    forecast_horizon = 20
    quantiles = [0.1, 0.5, 0.9]
    
    # y_true: (B, 20, 128, 128, 2) - Channel 0: NDVI, Channel 1: Landcover
    y_true_ndvi = tf.random.uniform((batch_size, forecast_horizon, h, w, 1))
    y_true_lc = tf.ones((batch_size, forecast_horizon, h, w, 1)) * 10.0 # Tree cover
    y_true = tf.concat([y_true_ndvi, y_true_lc], axis=-1)
    
    # y_pred: (B, NumQuantiles, 20, 128, 128, 1)
    y_pred = tf.random.uniform((batch_size, len(quantiles), forecast_horizon, h, w, 1))
    
    loss_fn = CombinedLoss(quantiles=quantiles, veg_weight=0.1)
    
    loss = loss_fn(y_true, y_pred)
    
    print(f"Loss value: {loss}")
    assert loss >= 0.0 # Loss might be slightly negative if VegScore > 1 (which is possible if NSE > 0.5), but usually we want to minimize it. 
    # Actually, VegetationScoreLoss returns 1/mean(NNSE) - 1. 
    # NNSE is in [0, 1] (since NSE <= 1). 
    # If NSE=1, NNSE=1, Loss=0. 
    # If NSE=0, NNSE=0.5, Loss=1.
    # If NSE < 0, NNSE < 0.5, Loss > 1.
    # So loss should be >= 0.
    
    print("CombinedLoss verified.")

if __name__ == "__main__":
    test_convformer_forward_pass()
    test_combined_loss()
