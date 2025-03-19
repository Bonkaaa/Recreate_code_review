from transformers import Trainer
import torch

class CustomSeq2SeqTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to unwrap model for `.generate()`."""
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module  # Unwrap model only for inference

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
