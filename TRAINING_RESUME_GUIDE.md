# Training Resume Guide for ConvFormer

## Overview

The ConvFormer training script has been updated to support **full checkpoint saving and loading**, enabling you to:
- Resume training after interruption
- Continue training with a different dataset (fine-tuning)
- Transfer learned weights to new training scenarios

## What's Saved in a Checkpoint

When using the new checkpoint system, the following are saved in `.keras` format:

1. **Model Architecture** - The complete ConvFormer structure
2. **Model Weights** - All learned parameters
3. **Optimizer State** - Adam optimizer's momentum and variance estimates
4. **Loss Function** - The CombinedLoss configuration
5. **Compilation Settings** - Learning rate and metrics

This is different from the old `.weights.h5` format which only saved the model weights.

## Basic Usage

### Starting New Training

```bash
python train_convformer.py \
    --train-dir /path/to/training/data \
    --epochs 200 \
    --batch-size 4 \
    --learning-rate 1e-4
```

This will create:
- `checkpoints/convformer_best.keras` - Best checkpoint during training
- `checkpoints/convformer_final.keras` - Final model after all epochs
- `logs/convformer/` - TensorBoard logs

### Resuming Training (Same Dataset)

If training was interrupted at epoch 100, resume with:

```bash
python train_convformer.py \
    --train-dir /path/to/training/data \
    --epochs 500 \
    --resume checkpoints/convformer_best.keras \
    --initial-epoch 100
```

**Important Notes:**
- `--initial-epoch` is mainly for display purposes and TensorBoard logging
- The optimizer state is fully restored, so training continues exactly where it left off
- Learning rate schedulers (ReduceLROnPlateau) will maintain their state

### Continuing with Different Dataset (Fine-tuning)

To fine-tune a pre-trained model on a new dataset:

```bash
python train_convformer.py \
    --train-dir /path/to/new/dataset \
    --val-dir /path/to/validation/dataset \
    --epochs 300 \
    --learning-rate 1e-5 \
    --resume checkpoints/convformer_best.keras \
    --initial-epoch 100 \
    --checkpoint-dir checkpoints_finetuned \
    --log-dir logs/convformer_finetuned
```

**Best Practices for Fine-tuning:**
- Use a **lower learning rate** (e.g., 1e-5 instead of 1e-4)
- Save to a **different checkpoint directory** to preserve original checkpoints
- Use a **different log directory** for separate TensorBoard tracking
- Consider using validation data to monitor overfitting

## Programmatic Usage

### Option 1: Resume with Full State

```python
from train_convformer import train_convformer

train_convformer(
    train_dir='/path/to/new/dataset',
    checkpoint_to_resume='checkpoints/convformer_best.keras',
    initial_epoch=100,
    epochs=300,
    learning_rate=1e-4  # Optimizer will use saved LR initially
)
```

### Option 2: Transfer Learning (Reset Optimizer)

For transfer learning where you want fresh optimizer state:

```python
import tensorflow as tf
from convformer import ConvFormer, CombinedLoss

# Load the saved model
saved_model = tf.keras.models.load_model('checkpoints/convformer_best.keras')

# Create new model and copy only weights
new_model = ConvFormer(forecast_horizon=20, quantiles=[0.1, 0.5, 0.9])
# Build model first
dummy_input = {...}  # See resume_training_example.py for structure
new_model(dummy_input)

# Transfer weights only
new_model.set_weights(saved_model.get_weights())

# Compile with fresh optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = CombinedLoss(quantiles=[0.1, 0.5, 0.9], veg_weight=0.1)
new_model.compile(optimizer=optimizer, loss=loss)

# Now train on new dataset with reset optimizer state
```

## Working with the Notebook

After the visualization cell in your notebook, you can continue training:

```python
# In your Jupyter notebook, after visualization

# Load the checkpoint (if not already loaded)
import tensorflow as tf

model = tf.keras.models.load_model(
    'checkpoints/convformer_best.keras',
    custom_objects={
        'ConvFormer': ConvFormer,
        'CombinedLoss': CombinedLoss,
        'QuantileLoss': QuantileLoss
    }
)

# Load new dataset
from greenearthnet_dataset import GreenEarthNetGenerator

new_train_dir = '/path/to/different/dataset'
generator = GreenEarthNetGenerator(new_train_dir)
new_dataset = generator.get_dataset().shuffle(100).batch(4).prefetch(tf.data.AUTOTUNE)

# Continue training
history = model.fit(
    new_dataset,
    epochs=50,  # Train for 50 more epochs
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/convformer_continued.keras',
            save_best_only=True,
            monitor='loss'
        )
    ]
)

# Visualize new results
# ... your visualization code ...
```

## Understanding initial_epoch

The `initial_epoch` parameter tells Keras which epoch number to start counting from:

- **Affects**: TensorBoard logging, progress display, callback monitoring
- **Does NOT affect**: Model weights, optimizer state (these come from the checkpoint)
- **Example**: If you stopped at epoch 100 and set `initial_epoch=100`, Keras will log epochs 100, 101, 102, etc.

**When to use:**
- Set to the last completed epoch when resuming
- Can be 0 if you don't care about epoch numbering continuity
- Important for proper TensorBoard visualization

## Monitoring with TensorBoard

View training progress:

```bash
tensorboard --logdir logs/convformer
```

When resuming training, use different log directories to compare runs:

```bash
tensorboard --logdir logs --reload_multifile=true
```

This shows:
- Original training: `logs/convformer/`
- Fine-tuned training: `logs/convformer_finetuned/`

## Troubleshooting

### "No checkpoints found"
- Verify the checkpoint path exists
- Use absolute paths or paths relative to where you run the script
- Check file extension is `.keras` not `.weights.h5`

### "Custom objects not found"
- Ensure you import ConvFormer, CombinedLoss, and QuantileLoss
- Pass them in the `custom_objects` dict when loading

### "Optimizer state mismatch"
- This can happen if you changed the model architecture
- Solution: Use transfer learning approach (copy weights only)

### Training restarts from scratch
- Make sure `checkpoint_to_resume` path is correct
- Check that the file exists with `os.path.exists(checkpoint_path)`
- Verify you're loading `.keras` files, not `.weights.h5`

## Examples

See `resume_training_example.py` for complete working examples of all three scenarios:
1. Resuming with same dataset
2. Fine-tuning with different dataset
3. Transfer learning with reset optimizer
