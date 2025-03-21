from transformers import Trainer
import torch

class CustomSeq2SeqTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to unwrap model for `.generate()`."""
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module  # Unwrap model only for inference

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def _issue_warnings_after_load(self, load_result):
        """Override to unwrap model after loading."""
        if isinstance(self.model._key_to_ignore_on_save, torch.nn.parallel.DistributedDataParallel):
            self.model._key_to_ignore_on_save = self.model._key_to_ignore_on_save.module
