# **Recreate Code Review (LLM Fine-Tuning with LoRA & QLoRA)**

This project focuses on fine-tuning the CodeT5 model for **Automated Code Review** tasks, and analyzing the performance differences between _LoRA_ and _QLoRA_ fine-tuning methods.<br>
It also includes a custom _Trainer (HuggingFace)_ implementation to validate results and compare against _LoRA_ and _QLoRA_ techniques.

### 📂 Repository Structure

```Recreate_code_review/
│
├── scripts/                # Shell scripts for running experiments
│   ├── args-parse.sh       # Parses CLI arguments for training
│   ├── test.sh             # Runs evaluation/testing
│   └── training.sh         # Runs training procedure
│
├── training/               # Core training & evaluation logic
│   ├── args_parse.py       # CLI arguments parser for Python
│   ├── checkpoint.py       # Model checkpoint handling
│   ├── evaluating.py       # Evaluation loop
│   ├── metrics.py          # Metrics calculation
│   ├── package-lock.json   # Dependency lockfile
│   ├── test.py             # Model testing script
│   ├── train.py            # Model training entry point
│   ├── utils.py            # Helper functions
│   ├── lora_config.py      # (LoRA branch) LoRA configuration
│   ├── bnb_config.py       # (QLoRA branch) QLoRA configuration
│   ├── customseq2Seq_trainer.py # (Trainer branch) Custom Seq2Seq Trainer
│   └── trainer.py          # (Trainer branch) Custom training loop
```

### 🌿 Branch Overview

* `main` → Base implementation with standard training and evaluation scripts.
* `LoRA` → Adds lora_config.py for LoRA-specific fine-tuning configuration.
* `QLoRA` → Adds bnb_config.py for QLoRA-specific fine-tuning configuration (using bitsandbytes).
* `Trainer` → Adds customseq2Seq_trainer.py and trainer.py for a custom training loop to validate LoRA & QLoRA results.

### ⚙️ Installation

```# Clone the repository
git clone https://github.com/Bonkaaa/Recreate_code_review.git
cd Recreate_code_review

# Install Python dependencies
pip install -r requirements.txt
```

_You may also need bitsandbytes, transformers, and peft for QLoRA and LoRA runs._

### 🚀 Usage

#### ⚠ Note:
The dataset used for fine-tuning is too large to be included in this repository.<br>
You will need to download or prepare your own dataset before running the scripts.

### 📊Experiment

This project compares the result implemented in **Automated Code Review**:

* `LoRA` fine-tuning
* `QLoRA` fine-tuning
* Custom `Trainer` fine-tuning

Metrics such as BLEU, Exact Match are computed in `training/metrics.py`.

[Link to Research Experiment](https://www.linkedin.com/posts/bonkaaa_lora-and-qlora-in-automated-code-review-activity-7324740857297829888-lnnG?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAFDP6O4Bsy0Aj2p-zAGG76OYuJHB587GrWg)

### 📝 Notes
`LoRA` and `QLoRA` configs are branch-specific.<br>
The `Trainer` branch serves as a baseline for evaluating the efficiency and validity of LoRA and QLoRA results.
