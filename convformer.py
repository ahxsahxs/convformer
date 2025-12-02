import tensorflow as tf
from tensorflow.keras import layers, models

@tf.keras.utils.register_keras_serializable(package="ConvFormer")
class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles, name='quantile_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        # y_true: (B, 20, 128, 128, 2) - Channel 0 is NDVI
        # y_pred: (B, NumQuantiles, 20, 128, 128, 1)
        
        y_true_ndvi = y_true[..., 0:1] # (B, 20, 128, 128, 1)
        
        # Expand y_true to match y_pred quantiles
        # (B, 1, 20, 128, 128, 1)
        y_true_exp = tf.expand_dims(y_true_ndvi, axis=1)
        
        # Mask NaNs
        mask = tf.logical_not(tf.math.is_nan(y_true_exp))
        y_true_safe = tf.where(mask, y_true_exp, 0.0)
        
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            q_pred = y_pred[:, i:i+1, ...] # (B, 1, 20, 128, 128, 1)
            error = y_true_safe - q_pred
            q_loss = tf.maximum(q * error, (q - 1) * error)
            loss += q_loss
            
        # Apply mask
        loss = tf.where(mask, loss, 0.0)
        
        # Average over valid pixels
        num_valid = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        return tf.reduce_sum(loss) / num_valid

    def get_config(self):
        config = super().get_config()
        config.update({'quantiles': self.quantiles})
        return config

@tf.keras.utils.register_keras_serializable(package="ConvFormer")
class VegetationScoreLoss(tf.keras.losses.Loss):
    """
    Vegetation Score Loss.
    
    Maximizes the Vegetation Score:
    VegScore = 2 - 1/mean(NNSE_veg)
    where NNSE = 1 / (2 - NSE)
    and NSE is Nash-Sutcliffe Efficiency on cloud-free vegetation pixels.
    
    Loss = 1 - VegScore (to minimize)
         = 1 - (2 - 1/mean(NNSE))
         = 1/mean(NNSE) - 1
    
    Vegetation classes (ESA WorldCover):
    10: Tree cover
    20: Shrubland
    30: Grassland
    """
    def __init__(self, name='vegetation_score_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        
    def call(self, y_true, y_pred):
        # y_true: (batch, 20, 128, 128, 2) [NDVI, Landcover]
        # y_pred: (batch, 20, 128, 128, 1)
        
        y_true_ndvi = y_true[..., 0:1]
        landcover = y_true[..., 1:2]
        
        # Mask for valid pixels (not NaN)
        valid_mask = tf.logical_not(tf.math.is_nan(y_true_ndvi))
        
        # Mask for vegetation pixels (10, 20, 30)
        # ESA WorldCover: 10=Trees, 20=Shrubland, 30=Grassland
        veg_mask = (landcover == 10) | (landcover == 20) | (landcover == 30)
        
        # Combined mask: Valid AND Vegetation
        mask = valid_mask & veg_mask
        
        # Replace NaNs with zeros for calculation
        y_true_safe = tf.where(valid_mask, y_true_ndvi, 0.0)
        
        # Count valid observations per pixel
        valid_count_per_pixel = tf.reduce_sum(tf.cast(valid_mask, tf.float32), axis=1, keepdims=True) # (B, 1, 128, 128, 1)
        
        # Sum of true values
        sum_true = tf.reduce_sum(y_true_safe, axis=1, keepdims=True)
        mean_true = tf.math.divide_no_nan(sum_true, valid_count_per_pixel)
        
        # Numerator: Sum of squared errors
        sse = tf.reduce_sum(tf.square(y_true_safe - y_pred) * tf.cast(valid_mask, tf.float32), axis=1, keepdims=True)
        
        # Denominator: Sum of squared deviations from mean
        sst = tf.reduce_sum(tf.square(y_true_safe - mean_true) * tf.cast(valid_mask, tf.float32), axis=1, keepdims=True)
        
        # NSE = 1 - SSE/SST
        epsilon = 1e-6
        nse = 1.0 - (sse / (sst + epsilon))
        
        # NNSE = 1 / (2 - NSE)
        nnse = 1.0 / (2.0 - nse)
        
        # Now average NNSE over vegetation pixels
        # veg_mask is (B, 20, 128, 128, 1). We need a spatial mask (B, 1, 128, 128, 1)
        # A pixel is vegetation if it is vegetation at any time step (it's static)
        spatial_veg_mask = veg_mask[:, 0:1, :, :, :] # (B, 1, 128, 128, 1)
        
        masked_nnse = nnse * tf.cast(spatial_veg_mask, tf.float32)
        
        sum_nnse = tf.reduce_sum(masked_nnse)
        count_veg = tf.reduce_sum(tf.cast(spatial_veg_mask, tf.float32))
        
        mean_nnse = tf.math.divide_no_nan(sum_nnse, count_veg)
        
        # Loss = 1/mean_nnse - 1
        loss = tf.where(
            count_veg > 0,
            (1.0 / (mean_nnse + epsilon)) - 1.0,
            0.0
        )
        
        return loss

@tf.keras.utils.register_keras_serializable(package="ConvFormer")
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles, veg_weight=0.1, name='combined_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantiles = quantiles
        self.veg_weight = veg_weight
        self.quantile_loss = QuantileLoss(quantiles)
        self.veg_loss = VegetationScoreLoss()
        
    def call(self, y_true, y_pred):
        # y_true: (B, 20, 128, 128, 2)
        # y_pred: (B, NumQuantiles, 20, 128, 128, 1)
        
        # 1. Quantile Loss
        q_loss = self.quantile_loss(y_true, y_pred)
        
        # 2. Vegetation Score Loss
        # Use the median prediction (usually the middle quantile) for the vegetation score
        # Assuming quantiles are sorted, median is at index len(quantiles)//2
        median_idx = len(self.quantiles) // 2
        
        # Extract median prediction: (B, NumQuantiles, 20, 128, 128, 1) -> (B, 20, 128, 128, 1)
        # Slicing with integer index removes the dimension
        y_pred_median = y_pred[:, median_idx, ...]
        
        v_loss = self.veg_loss(y_true, y_pred_median)
        
        return q_loss + self.veg_weight * v_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'quantiles': self.quantiles,
            'veg_weight': self.veg_weight
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="ConvFormer")
class QuantileRegressionHead(layers.Layer):
    def __init__(self, forecast_horizon, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.quantile_heads = [
            tf.keras.Sequential([
                layers.Conv2D(256, 3, padding='same', activation='relu'),
                layers.Conv2D(128, 3, padding='same', activation='relu'),
                layers.Conv2D(forecast_horizon, 1) 
            ], name=f'q_head_{i}')
            for i in range(self.num_quantiles)
        ]

    def call(self, inputs):
        outputs = []
        for head in self.quantile_heads:
            pred = head(inputs)
            pred = tf.transpose(pred, perm=[0, 3, 1, 2]) # (B, 20, 128, 128)
            pred = tf.expand_dims(pred, axis=-1) # (B, 20, 128, 128, 1)
            outputs.append(pred)
            
        return tf.stack(outputs, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "forecast_horizon": self.forecast_horizon,
            "quantiles": self.quantiles,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="ConvFormer")
class ConvFormer(tf.keras.Model):
    def __init__(self, forecast_horizon=20, quantiles=[0.1, 0.5, 0.9], **kwargs):
        super().__init__(**kwargs)
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        
        # --- Encoder ---
        self.spatial_encoder = tf.keras.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2), 
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2), 
            layers.Conv2D(128, 3, padding='same', activation='relu')
        ], name='spatial_encoder')
        
        self.td_spatial = layers.TimeDistributed(self.spatial_encoder)
        
        self.conv_lstm = layers.ConvLSTM2D(
            filters=128, kernel_size=3, padding='same', return_sequences=False
        )
        
        # --- Fusion ---
        self.weather_mlp = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu')
        ], name='weather_mlp')
        
        self.dem_encoder = tf.keras.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2), 
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2), 
            layers.Conv2D(128, 3, padding='same', activation='relu')
        ], name='dem_encoder')
        
        self.fusion_conv = layers.Conv2D(256, 3, padding='same', activation='relu')
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.UpSampling2D(2), 
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.UpSampling2D(2), 
            layers.Conv2D(64, 3, padding='same', activation='relu')
        ], name='decoder')
        
        # --- Head ---
        self.head = QuantileRegressionHead(forecast_horizon, quantiles)

    def call(self, inputs):
        if isinstance(inputs, dict):
            sentinel2 = inputs['sentinel2']
            weather = inputs['weather']
            dem = inputs['dem']
        else:
            sentinel2, weather, dem = inputs
            
        x_spatial = self.td_spatial(sentinel2)
        x_temporal = self.conv_lstm(x_spatial)
        
        w_feat = tf.reduce_mean(weather, axis=1) 
        w_emb = self.weather_mlp(w_feat) 
        w_emb = tf.reshape(w_emb, (-1, 1, 1, 128))
        w_emb = tf.tile(w_emb, [1, 32, 32, 1]) 
        
        d_emb = self.dem_encoder(dem)
        
        fused = layers.concatenate([x_temporal, w_emb, d_emb], axis=-1)
        fused = self.fusion_conv(fused) 
        
        decoded = self.decoder(fused)
        
        outputs = self.head(decoded)
        
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "forecast_horizon": self.forecast_horizon,
            "quantiles": self.quantiles,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
