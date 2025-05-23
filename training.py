from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from generator import Train_Discriminator,Train_GAN
from model import SharpGan
import numpy as np
from tensorflow.keras.callbacks import Callback
import os # Keep os for potential environment variables, but not for path joining
from pathlib import Path
import cv2
from typing import Dict, Optional, List # Optional for logs=None, List for g_loss

class Monitor(Callback):
    """A Keras Callback to capture and store training logs."""
    def __init__(self) -> None:
        """Initializes the Monitor callback with default log values."""
        super().__init__() # Call superclass constructor
        self.logs: Dict[str, float] = {'loss':1.0,
                                       'discriminator_loss':1.0,
                                       'out_generator_loss':1.0}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """Called at the end of an epoch to update the stored logs."""
        if logs is not None:
            self.logs = logs

# Create the model instance globally as it's used by train()
# Learning rates can be made configurable in train() later if needed
model = SharpGan(d_lr=1e-3, g_lr=2e-3) # type: ignore
model.D.summary()
model.G.summary()
model.GAN.summary()

def train(nb_epoch: int, current_cycle: int = 1, batch_size: int = 8, base_data_dir: str = "ground_truth", simulated_data_dir: str = "simulated", output_base_dir: str = "run_outputs") -> None:
    """Trains the SharpGAN model for a specified number of epochs and configurations."""
    BATCH_SIZE: int = batch_size
    CURRENT_CYCLE: int = current_cycle
    steps: int = int(2800 / BATCH_SIZE) # Assuming 2800 is a fixed dataset size

    # --- Pathlib Implementation ---
    base_output_path = Path(output_base_dir)
    cycle_output_dir = base_output_path / str(CURRENT_CYCLE)
    cycle_output_dir.mkdir(parents=True, exist_ok=True)

    # Model weights paths
    if CURRENT_CYCLE > 1:
        gan_load_weights_path = base_output_path / str(CURRENT_CYCLE - 1) / f'GAN_trained_{CURRENT_CYCLE - 1}.h5'
        if gan_load_weights_path.exists():
            print(f"LOADING OLD WEIGHTS from {gan_load_weights_path}")
            model.GAN.load_weights(str(gan_load_weights_path))
            print('*************************************')
        else:
            print(f"Old weights path not found: {gan_load_weights_path}")

    ckpt_g_path = cycle_output_dir / f'Generator_{CURRENT_CYCLE}.h5'
    gan_save_path = cycle_output_dir / f'GAN_trained_{CURRENT_CYCLE}.h5'

    # CSV log path
    train_result_path = cycle_output_dir / f'train_{CURRENT_CYCLE}.csv'

    # Image saving paths
    image_cycle_dir = cycle_output_dir / 'images'
    image_cycle_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Generator Instantiation ---
    train_D_fake = Train_Discriminator(model.G, base_data_dir, simulated_data_dir, BATCH_SIZE, True)
    train_D = Train_Discriminator(model.G, base_data_dir, simulated_data_dir, BATCH_SIZE, False)
    train_GAN = Train_GAN(base_data_dir, simulated_data_dir, BATCH_SIZE)

    # Monitor for Discriminator and GAN
    monitor_D_real = Monitor()
    monitor_D = Monitor()
    monitor_GAN = Monitor()

    # Model Checkpoint
    ckpt_g = ModelCheckpoint(str(ckpt_g_path), monitor='out_generator_loss', save_best_only=False)

    epoch_D = 1
    epoch_G = 1
    
    # Use line buffering (buffering=1) for the CSV
    with open(str(train_result_path), 'w', buffering=1) as train_result:
        train_result.write('epoch,d_loss,SSIM+L1,g_loss\n')
        
        for i in range(nb_epoch + 1): # Loop from 0 to nb_epoch inclusive
            print(f'TRAINING EPOCH {i + 1}/{nb_epoch}')

            # Evaluate current model state (epoch 0 is initial state)
            x_val_d, y_val_d = train_D_fake.get_validation(20)
            d_loss = model.D.evaluate(x_val_d, y_val_d, BATCH_SIZE, verbose=0)

            x_val_g, y_val_g = train_GAN.get_validation(20)
            g_loss = model.GAN.evaluate(x_val_g, y_val_g, BATCH_SIZE, verbose=0)
            
            # Log losses. Ensure g_loss has at least 3 elements if you are accessing g_loss[1] and g_loss[2]
            # For GAN, loss is typically [total_loss, generator_output_loss, discriminator_output_loss]
            # If self.loss_G is the only loss for generator output, then g_loss[1] is that.
            # And g_loss[2] would be the discriminator's loss component from GAN.
            # The exact indices depend on model.GAN.compile(loss=...) structure.
            # Assuming g_loss[0] is total, g_loss[1] is G_loss (e.g. SSIM+L1), g_loss[2] is D_loss_on_G_output
            current_d_loss_val: float = d_loss # d_loss is already float from model.evaluate
            
            # g_loss can be a single float or a list of floats
            g_loss_val: float
            g_loss_gan_component: float = -1.0 # Or some other placeholder for N/A

            if isinstance(g_loss, list):
                if len(g_loss) > 2:
                    g_loss_val = g_loss[1]
                    g_loss_gan_component = g_loss[2]
                elif len(g_loss) > 1:
                    g_loss_val = g_loss[1]
                else: # Should not happen if model compiled correctly
                    g_loss_val = g_loss[0] if g_loss else -1.0 
            else: # single float
                g_loss_val = g_loss
            
            if g_loss_gan_component != -1.0:
                train_result.write(f'{i},{current_d_loss_val},{g_loss_val},{g_loss_gan_component}\n')
            else:
                train_result.write(f'{i},{current_d_loss_val},{g_loss_val},N/A\n')


            if i != 0 and i % 2 == 0:
                # Draw prediction after 2 epochs
                prediction = model.G.predict(x_val_g) # Use validation data for consistent visualization
                epoch_image_dir = image_cycle_dir / f'epoch_{i}'
                epoch_image_dir.mkdir(parents=True, exist_ok=True)
                
                for img_idx in range(20): # Iterate up to 20 or len(prediction)
                    img_save_path = epoch_image_dir / f'{img_idx + 1}.jpg'
                    pred_image = prediction[img_idx, :, :, :]
                    pred_image = np.clip(pred_image * 255, 0, 255)
                    cv2.imwrite(str(img_save_path), pred_image)
                
                # Also save weight for that epoch
                epoch_weights_save_path = epoch_image_dir / 'saved_weights.h5'
                model.GAN.save(str(epoch_weights_save_path))

            if i == nb_epoch: # Stop after logging and saving images for the final epoch
                break

            print(f"Current D_loss: {current_d_loss_val}")
            print(f"Current G_loss (components): {g_loss_val}, {g_loss_gan_component if g_loss_gan_component != -1.0 else 'N/A'}")

            # Pre-train the Discriminator
            print("Train the Discriminator on real images")
            model.D.fit(generator=train_D, # type: ignore
                                  steps_per_epoch=steps,
                                  epochs=epoch_D, # train for 1 sub-epoch
                                  validation_data=train_D.get_validation(20),
                                  callbacks=[monitor_D_real],
                                  max_queue_size=5)
            print("Train the Discriminator on Fake images")
            model.D.fit(generator=train_D_fake, # type: ignore
                                  steps_per_epoch=steps,
                                  epochs=epoch_D, # train for 1 sub-epoch
                                  validation_data=train_D_fake.get_validation(20),
                                  callbacks=[monitor_D],
                                  max_queue_size=5)

            print("AFTER DISCRIMINATOR TRAIN")
            x_val_d_after, y_val_d_after = train_D_fake.get_validation(20) # Fresh validation data
            d_loss_after: float = model.D.evaluate(x_val_d_after, y_val_d_after, BATCH_SIZE, verbose=0) # type: ignore
            print(f'Discriminator loss on fake images after training: {d_loss_after}')

            # Train the Generator through GAN
            print("Train the Generator/GAN")
            model.GAN.fit(generator=train_GAN, # type: ignore
                                    steps_per_epoch=steps,
                                    epochs=epoch_G, # train for 1 sub-epoch
                                    validation_data=train_GAN.get_validation(20),
                                    callbacks=[ckpt_g, monitor_GAN], # ckpt_g uses the pathlib path
                                    max_queue_size=5)
            
            model.GAN.save(str(gan_save_path)) # Save main model

if __name__ == "__main__":
    # Example of how to call the train function
    # These values can be configured via argparse or other means
    train(nb_epoch=20, current_cycle=1, batch_size=8)