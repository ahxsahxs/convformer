import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
import tensorflow as tf

class GreenEarthNetGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, "**/*.nc"), recursive=True))
        self.input_days = 50
        self.target_days = 100
        self.s2_bands = ['s2_B02', 's2_B03', 's2_B04', 's2_B8A']
        self.eobs_vars = ['eobs_tg', 'eobs_fg', 'eobs_hu', 'eobs_pp', 'eobs_qq', 'eobs_rr', 'eobs_tn', 'eobs_tx']
        
        # Normalization stats from en21x_data.py
        self.eobs_stats = {
            'eobs_tg': {'mean': 8.9066, 'std': 9.7562},
            'eobs_fg': {'mean': 2.7329, 'std': 1.4870},
            'eobs_hu': {'mean': 77.5444, 'std': 13.5114},
            'eobs_pp': {'mean': 1014.3310, 'std': 10.2626},
            'eobs_qq': {'mean': 126.4792, 'std': 97.0552},
            'eobs_rr': {'mean': 1.7713, 'std': 4.1480},
            'eobs_tn': {'mean': 4.7707, 'std': 9.0450},
            'eobs_tx': {'mean': 13.5680, 'std': 11.0820}
        }
        
        self.dem_vars = ['alos_dem', 'cop_dem', 'nasa_dem']
        
    def compute_ndvi(self, red, nir):
        # NDVI = (NIR - Red) / (NIR + Red)
        denominator = nir + red
        ndvi = np.divide(nir - red, denominator, out=np.zeros_like(denominator), where=denominator!=0)
        # Normalize to [0, 1] as per methodology: [-1, 1] -> [0, 1]
        ndvi_norm = (ndvi + 1) / 2
        return np.clip(ndvi_norm, 0, 1)

    def normalize_band(self, band_data):
        # Simple min-max normalization or percentile based?
        # The previous code used percentile. Let's stick to a robust normalization or just raw for now if not specified.
        # User request didn't specify normalization, but usually it's good practice.
        # However, for a generator "to be used later", raw values might be preferred unless specified.
        # But previous dataset.py had normalization.
        # Let's keep it simple: Raw values for now, or maybe 0-1 if we know the range.
        # Sentinel-2 is usually 0-10000.
        return np.clip(band_data / 10000.0, 0, 1)

    def __call__(self):
        for file_path in self.files:
            try:
                with xr.open_dataset(file_path) as ds:
                    # Check if we have enough time steps
                    if len(ds.time) < self.input_days + self.target_days:
                        continue

                    # --- Inputs (First 50 days) ---
                    input_slice = slice(0, self.input_days)
                    
                    # 1. Sentinel-2 Bands
                    s2_data = []
                    for band in self.s2_bands:
                        b_data = ds[band].isel(time=input_slice).values
                        # Normalize? Let's do simple division by 10000 for S2
                        s2_data.append(np.clip(b_data / 10000.0, 0, 1))
                    
                    # (50, 128, 128, 4)
                    sentinel2 = np.stack(s2_data, axis=-1)
                    
                    # Check for NaNs in Sentinel-2
                    # If any band is NaN, the pixel is missing/invalid
                    s2_nans = np.isnan(sentinel2).any(axis=-1, keepdims=True) # (50, 128, 128, 1)
                    
                    # Replace NaNs with 0.0
                    sentinel2 = np.nan_to_num(sentinel2, nan=0.0)

                    # 2. Cloud Mask
                    # mask > 0 means cloud/shadow etc.
                    mask = ds['s2_mask'].isel(time=input_slice).values
                    s2_mask = (mask > 0).astype(np.float32)
                    s2_mask = np.expand_dims(s2_mask, axis=-1) # (50, 128, 128, 1)
                    
                    # Update mask to include NaNs (missing data)
                    # If s2_nans is True, s2_mask should be 1 (masked)
                    s2_mask = np.maximum(s2_mask, s2_nans.astype(np.float32))
                
                    # 3. Weather (E-OBS)
                    weather_data = []
                    for var in self.eobs_vars:
                        w_data = ds[var].isel(time=input_slice).values
                        # Normalize
                        stats = self.eobs_stats[var]
                        w_data = (w_data - stats['mean']) / stats['std']
                        weather_data.append(w_data)                    
                    
                    # (50, 8)
                    weather = np.stack(weather_data, axis=-1)
                    # Handle NaNs in weather if any (though usually E-OBS is complete or interpolated)
                    weather = np.nan_to_num(weather, nan=0.0)

                    # 4. DEM
                    dem_data = []
                    for var in self.dem_vars:
                        d_data = ds[var].values # (lat, lon)
                        # Normalize by dividing by 500
                        d_data = d_data / 500.0
                        dem_data.append(d_data)
                    
                    # (128, 128, 3)
                    dem = np.stack(dem_data, axis=-1)
                    # Handle NaNs in DEM
                    dem = np.nan_to_num(dem, nan=0.0)

                    # 5. Geomorphology
                    geom = ds['geom_cls'].values # (lat, lon)
                    geomorphology = np.expand_dims(geom, axis=-1) # (128, 128, 1)
                    geomorphology = np.nan_to_num(geomorphology, nan=0.0)
                    
                    # 6. Landcover (ESA WorldCover)
                    lc = ds['esawc_lc'].values # (128, 128)
                    
                    # 7. Time
                    times = ds.time.isel(time=input_slice).values
                    ts = pd.to_datetime(times)
                    
                    # Cyclical features for Day of Year
                    doy = ts.dayofyear
                    doy_sin = np.sin(2 * np.pi * doy / 366.0)
                    doy_cos = np.cos(2 * np.pi * doy / 366.0)
                    
                    # Normalize Year (approximate, assuming data is recent)
                    # Let's map 2017-2021 to roughly [-1, 1] or [0, 1]
                    # 2017 is start, 2021 is end.
                    year_norm = (ts.year - 2019) / 2.0 
                    
                    time_feats = np.stack([year_norm, doy_sin, doy_cos], axis=-1).astype(np.float32) # (50, 3)

                    x = {
                        'sentinel2': sentinel2.astype(np.float32),
                        's2_mask': s2_mask.astype(np.float32),
                        'weather': weather.astype(np.float32),
                        'dem': dem.astype(np.float32),
                        'geomorphology': geomorphology.astype(np.float32),
                        'time': time_feats
                    }

                    # --- Targets (Next 100 days) ---
                    target_slice = slice(self.input_days+4, self.input_days + self.target_days, 5)
                    
                    red_t = ds['s2_B04'].isel(time=target_slice).values
                    nir_t = ds['s2_B8A'].isel(time=target_slice).values
                    
                    ndvi_t = self.compute_ndvi(red_t, nir_t)
                    
                    # Apply cloud mask
                    mask_t = ds['s2_mask'].isel(time=target_slice).values
                    
                    # Use np.where for safe broadcasting/masking
                    ndvi_t = np.where(mask_t > 0, np.nan, ndvi_t)
                        
                    avail_t = ds['s2_avail'].isel(time=target_slice).values
                    # avail_t is (time,)
                    # Broadcast to (time, 128, 128)
                    avail_t = avail_t[:, None, None]
                    ndvi_t = np.where(avail_t == 0, np.nan, ndvi_t)

                    # Prepare y with shape (20, 128, 128, 2)
                    # Channel 0: NDVI
                    # Channel 1: Landcover (repeated)
                    
                    # Expand NDVI to (20, 128, 128, 1)
                    ndvi_t = np.expand_dims(ndvi_t, axis=-1)
                    
                    # Prepare Landcover (1, 128, 128, 1) -> (20, 128, 128, 1)
                    lc_expanded = np.expand_dims(np.expand_dims(lc, axis=0), axis=-1)
                    lc_t = np.tile(lc_expanded, (ndvi_t.shape[0], 1, 1, 1))
                    
                    y = np.concatenate([ndvi_t, lc_t], axis=-1).astype(np.float32)

                    yield x, y

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    def get_dataset(self) -> tf.data.Dataset:
        # Define output signature
        output_signature = (
            {
                'sentinel2': tf.TensorSpec(shape=(50, 128, 128, 4), dtype=tf.float32),
                's2_mask': tf.TensorSpec(shape=(50, 128, 128, 1), dtype=tf.float32),
                'weather': tf.TensorSpec(shape=(50, 8), dtype=tf.float32),
                'dem': tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
                'geomorphology': tf.TensorSpec(shape=(128, 128, 1), dtype=tf.float32),
                'time': tf.TensorSpec(shape=(50, 3), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(20, 128, 128, 2), dtype=tf.float32)
        )
        
        return tf.data.Dataset.from_generator(
            self.__call__,
            output_signature=output_signature
        )
