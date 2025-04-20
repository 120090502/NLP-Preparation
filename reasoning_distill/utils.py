import os
import logging
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch

class TrainingVisualizer:
    """A utility class for visualizing training metrics and saving loss plots."""
    
    def __init__(self, log_dir):
        """
        Initialize the TrainingVisualizer.
        
        Args:
            log_dir (str): Directory to store TensorBoard logs and loss plots.
        """
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.logger.info(f"Initialized TensorBoard writer at {self.log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard writer: {str(e)}")
            raise
            
    def plot_loss(self, output_dir):
        """
        Plot the training loss and save the plot to the output directory.
        
        Args:
            output_dir (str): Directory to save the loss plot.
            
        Returns:
            str: Path to the saved loss plot file.
        """
        try:
            # This assumes loss data is logged to TensorBoard during training
            # In a real scenario, you might need to load loss data from a file or trainer
            # For simplicity, we'll create a placeholder plot
            plot_path = os.path.join(output_dir, "training_loss.png")
            
            # Placeholder: Create a simple plot (actual implementation depends on data source)
            plt.figure(figsize=(10, 6))
            plt.title("Training Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.plot([0], [0], label="Loss (placeholder)")  # Placeholder data
            plt.legend()
            
            # Save the plot
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Saved loss plot to {plot_path}")
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Failed to save loss plot: {str(e)}")
            raise
            
    def close(self):
        """Close the TensorBoard writer to flush all data."""
        try:
            self.writer.close()
            self.logger.info("Closed TensorBoard writer")
        except Exception as e:
            self.logger.error(f"Failed to close TensorBoard writer: {str(e)}")
            raise

class ModelSaver:
    """A utility class for saving trained models and tokenizers."""
    
    @staticmethod
    def save(model, tokenizer, output_dir):
        """
        Save the model and tokenizer to the specified output directory.
        
        Args:
            model: The trained model to save.
            tokenizer: The tokenizer to save.
            output_dir (str): Directory to save the model and tokenizer.
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the model
            model_save_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved model state dict to {model_save_path}")
            
            # Save the model's configuration
            model.config.save_pretrained(output_dir)
            logger.info(f"Saved model configuration to {output_dir}")
            
            # Save the tokenizer
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved tokenizer to {output_dir}")
            
            # Save the PEFT (LoRA) weights separately if applicable
            if hasattr(model, "peft_config"):
                model.save_pretrained(output_dir)
                logger.info(f"Saved PEFT (LoRA) weights to {output_dir}")
                
        except Exception as e:
            logger.error(f"Failed to save model or tokenizer: {str(e)}")
            raise