import os
import sys
import argparse
import tensorflow as tf
# Disable GPU to avoid "No DNN in stream executor" error
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from convformer import ConvFormer, QuantileLoss, CombinedLoss
from greenearthnet_dataset import GreenEarthNetGenerator

def train_convformer(
    train_dir,
    val_dir=None,
    batch_size=4,
    epochs=200,
    learning_rate=1e-4,
    checkpoint_dir='checkpoints',
    log_dir='logs/convformer',
    quantiles=[0.1, 0.5, 0.9],
    checkpoint_to_resume=None,
    initial_epoch=0
):
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Loading data from {train_dir}...")
    
    # 1. Dataset
    generator = GreenEarthNetGenerator(train_dir)
    train_dataset = generator.get_dataset()
    
    # Shuffle and Batch
    train_dataset = train_dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = None
    if val_dir and os.path.exists(val_dir):
        print(f"Loading validation data from {val_dir}...")
        val_generator = GreenEarthNetGenerator(val_dir)
        val_dataset = val_generator.get_dataset()
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 2. Model - Either load from checkpoint or create new
    if checkpoint_to_resume and os.path.exists(checkpoint_to_resume):
        print(f"Loading model from checkpoint: {checkpoint_to_resume}")
        # Load the full model including optimizer state
        model = tf.keras.models.load_model(
            checkpoint_to_resume,
            custom_objects={
                'ConvFormer': ConvFormer,
                'CombinedLoss': CombinedLoss,
                'QuantileLoss': QuantileLoss
            }
        )
        print("Model loaded successfully. Resuming training...")
        model.summary()
    else:
        if checkpoint_to_resume:
            print(f"Warning: Checkpoint {checkpoint_to_resume} not found. Creating new model...")
        
        print("Creating ConvFormer model...")
        model = ConvFormer(forecast_horizon=20, quantiles=quantiles)
        
        # Build model to print summary
        # Create dummy input to build the model
        dummy_input = {
            'sentinel2': tf.zeros((1, 50, 128, 128, 4)),
            's2_mask': tf.zeros((1, 50, 128, 128, 1)),
            'weather': tf.zeros((1, 50, 8)),
            'dem': tf.zeros((1, 128, 128, 3)),
            'geomorphology': tf.zeros((1, 128, 128, 1)),
            'time': tf.zeros((1, 50, 3))
        }
        model(dummy_input)
        model.summary()
        
        # 3. Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = CombinedLoss(quantiles=quantiles, veg_weight=0.1)
        
        model.compile(optimizer=optimizer, loss=loss)
    
    # 4. Callbacks
    # Save full model (not just weights) to preserve optimizer state
    checkpoint_path = os.path.join(checkpoint_dir, 'convformer_best.keras')
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='loss',
            save_best_only=True,
            save_weights_only=False,  # Save full model including optimizer
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=25,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    # 5. Train
    print(f"Starting training from epoch {initial_epoch}...")
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
        return model
    
    # Save final model (full model with optimizer state)
    final_model_path = os.path.join(checkpoint_dir, 'convformer_final.keras')
    print(f"Saving final model to {final_model_path}...")
    model.save(final_model_path)
    
    print("Training complete.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ConvFormer model')
    parser.add_argument(
        '--train-dir',
        type=str,
        default='/home/me/workspace/probformer/data/greenearthnet/traint_test',
        help='Directory containing training .nc files'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default=None,
        help='Directory containing validation .nc files'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='Maximum number of epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/convformer',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from (e.g., checkpoints/convformer_best.keras)'
    )
    parser.add_argument(
        '--initial-epoch',
        type=int,
        default=0,
        help='Starting epoch number (useful when resuming training)'
    )
    
    args = parser.parse_args()
    
    # Use env var if argument is default and env var is set? 
    # Or just rely on args. The user specified the path in the prompt.
    
    train_convformer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        checkpoint_to_resume=args.resume,
        initial_epoch=args.initial_epoch
    )
