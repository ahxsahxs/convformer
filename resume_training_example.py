"""
Example: How to Resume Training with a Different Dataset

This script demonstrates how to continue training the ConvFormer model
using a saved checkpoint and a different dataset.
"""

from train_convformer import train_convformer

# Option 1: Resume training with the same dataset
# This will continue from where training stopped
def resume_same_dataset():
    train_convformer(
        train_dir='/home/me/workspace/probformer/data/greenearthnet/train',
        val_dir=None,
        batch_size=4,
        epochs=500,  # Continue to epoch 500
        learning_rate=1e-4,
        checkpoint_dir='checkpoints',
        log_dir='logs/convformer',
        checkpoint_to_resume='checkpoints/convformer_best.keras',  # Load saved checkpoint
        initial_epoch=100  # Start from epoch 100 (if you stopped at epoch 100)
    )

# Option 2: Resume training with a different dataset
# Use a checkpoint trained on one dataset and continue training on another
def resume_different_dataset():
    train_convformer(
        train_dir='/home/me/workspace/probformer/data/greenearthnet/new_dataset',  # Different dataset
        val_dir='/home/me/workspace/probformer/data/greenearthnet/val',
        batch_size=4,
        epochs=300,  # Train for 200 more epochs (from 100 to 300)
        learning_rate=1e-5,  # Often use lower learning rate when fine-tuning
        checkpoint_dir='checkpoints_finetuned',  # Save to a different directory
        log_dir='logs/convformer_finetuned',  # Different log directory
        checkpoint_to_resume='checkpoints/convformer_best.keras',  # Load previous checkpoint
        initial_epoch=100  # Starting epoch number (mainly for display purposes)
    )

# Option 3: Load only weights (not optimizer state) for transfer learning
# This is useful when you want a fresh start with optimizer but keep learned weights
def transfer_learning():
    import tensorflow as tf
    from convformer import ConvFormer, CombinedLoss
    from greenearthnet_dataset import GreenEarthNetGenerator
    import os
    
    # Create new model
    model = ConvFormer(forecast_horizon=20, quantiles=[0.1, 0.5, 0.9])
    
    # Build the model
    dummy_input = {
        'sentinel2': tf.zeros((1, 50, 128, 128, 4)),
        's2_mask': tf.zeros((1, 50, 128, 128, 1)),
        'weather': tf.zeros((1, 50, 8)),
        'dem': tf.zeros((1, 128, 128, 3)),
        'geomorphology': tf.zeros((1, 128, 128, 1)),
        'time': tf.zeros((1, 50, 3))
    }
    model(dummy_input)
    
    # Load ONLY the weights from a saved full model
    saved_model = tf.keras.models.load_model('checkpoints/convformer_best.keras')
    model.set_weights(saved_model.get_weights())
    print("Weights loaded from checkpoint!")
    
    # Compile with fresh optimizer (optimizer state is reset)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = CombinedLoss(quantiles=[0.1, 0.5, 0.9], veg_weight=0.1)
    model.compile(optimizer=optimizer, loss=loss)
    
    # Load new dataset
    train_dir = '/home/me/workspace/probformer/data/greenearthnet/new_dataset'
    generator = GreenEarthNetGenerator(train_dir)
    train_dataset = generator.get_dataset().shuffle(100).batch(4).prefetch(tf.data.AUTOTUNE)
    
    # Train with new dataset
    os.makedirs('checkpoints_transfer', exist_ok=True)
    checkpoint_path = 'checkpoints_transfer/convformer_best.keras'
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir='logs/convformer_transfer')
    ]
    
    history = model.fit(
        train_dataset,
        epochs=200,
        callbacks=callbacks
    )
    
    return model

if __name__ == "__main__":
    # Choose which option to run
    
    # Uncomment the one you want to use:
    
    # resume_same_dataset()
    # resume_different_dataset()
    # transfer_learning()
    
    print("Please uncomment the desired function in the __main__ block to run.")
