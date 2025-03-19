from typing import Dict, Union, Any, Optional, List, Tuple

import torch
from torch import nn
from transformers import Seq2SeqTrainer

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ):
        if hasattr(model, 'module'):
            generation_inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "labels"}
            generated_tokens = model.module.generate(**generation_inputs)
            # Rest of the original prediction_step implementation
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)