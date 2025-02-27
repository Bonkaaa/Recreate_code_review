from peft import LoraConfig

def get_lora_config():
    # Define LoRA configuration
    return LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1"],  # Layers to apply LoRA
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

if __name__ == '__main__':
    pass