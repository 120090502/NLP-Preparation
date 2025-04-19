import os
import json
import logging
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        test_dataloader=None,
        save_dir="./ckpt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=5e-5,
        scheduler_type="cosine",
        num_epochs=10,
    ):
        """
        A reusable Trainer class for training, validation, and testing of models.

        Args:
            model (torch.nn.Module): Model to be trained (e.g., from transformers).
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for preprocessing data.
            train_dataloader (torch.utils.data.DataLoader): Training data loader.
            val_dataloader (torch.utils.data.DataLoader): Validation data loader.
            test_dataloader (torch.utils.data.DataLoader, optional): Test data loader. Defaults to None.
            save_dir (str, optional): Directory for saving checkpoints and logs. Defaults to "./ckpt".
            device (str, optional): Device to use for training. Defaults to "cuda" if available.
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 5e-5.
            scheduler_type (str, optional): Learning rate scheduler type ("cosine" or "linear"). Defaults to "cosine".
            num_epochs (int, optional): Number of training epochs. Defaults to 10.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.save_dir = save_dir
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.num_epochs = num_epochs

        os.makedirs(self.save_dir, exist_ok=True)
        self._setup_logging()

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        if self.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        else:
            self.scheduler = None

        self.model.to(self.device)

    def _setup_logging(self):
        """Sets up logging to file and console."""
        log_file = os.path.join(self.save_dir, "training.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger("").addHandler(console)

    def train_epoch(self):
        """Performs one epoch of training."""
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_dataloader)

    def evaluate(self, dataloader):
        """Evaluates the model."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_loss += loss.item()
        return total_loss / len(dataloader)

    def save_checkpoint(self, epoch):
        """Saves model checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"epoch_{epoch+1}.pt")
        self.model.save_pretrained(checkpoint_path)
        logging.info(f"Model checkpoint saved to {checkpoint_path}")

    def save_summary(self, train_loss_list, val_loss_list, test_loss=None):
        """Saves training summary to JSON."""
        summary = {
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "test_loss": test_loss,
        }
        summary_path = os.path.join(self.save_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        logging.info(f"Training summary saved to {summary_path}")

    def train(self):
        """Main training loop."""
        train_loss_list, val_loss_list = [], []

        for epoch in range(self.num_epochs):
            logging.info(f"Starting epoch {epoch+1}/{self.num_epochs}...")
            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.val_dataloader)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            logging.info(f"Epoch {epoch+1} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save model checkpoint
            self.save_checkpoint(epoch)

            # Step the scheduler if used
            if self.scheduler:
                self.scheduler.step()

        # Final evaluation on test set if available
        test_loss = None
        if self.test_dataloader:
            test_loss = self.evaluate(self.test_dataloader)
            logging.info(f"Final Test Loss: {test_loss:.4f}")

        # Save training summary
        self.save_summary(train_loss_list, val_loss_list, test_loss)

