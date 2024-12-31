# TabTransformer with Masked Pretraining: Unlocking Tabular Data Potential 🚀

Welcome to the repository for **TabTransformer with Masked Pretraining**, which should be the public example
demonstrating the training TabTransformers using Masked Language Modelling-style objectives.


---

## 🌟 Features

- **Domain Agnostic**: Supports diverse domains with minimal customization.
- **Extensible Framework**: Easily adapt the framework for new datasets and use cases.
- **Weights & Biases Integration**: Leverage Weights & Biases (WandB) for experiment tracking, model performance
  visualization, and hyperparameter optimization.
- **PyTorch Lightning Support**: Utilizes the PyTorch Lightning framework to simplify training loops, logging, and model
  checkpointing while ensuring scalability and reproducibility.

---

## 📂 Repository Structure

```plaintext
├── tabular_datamodule.py   # Data Preprocessing and dataloading 
├── tabular_module.py       # Implementation of TabTransformer architecture
├── train.py                # Training script with CLI support
├── pyproject.toml          # Poetry configuration file
```

Overall the repository is designed to be very hackable and the code should be pretty self-explanatory.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Poetry for dependency management

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/charitarthchugh/masked-tabtransformers-lightning.git
   cd masked-tabtransformers-lightning
   ```

2. **Install dependencies with Poetry**:
   ```bash
   poetry install
   ```
   *If you don't want to use Poetry, setting up PyTorch in a virtual environment and doing a pip install of other
   dependencies should work without issues*

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

---

## 🏋️ Training

### Command-Line Interface

The training script supports multiple configuration options. Use the following command to start training:

```bash
python train.py \
  --train-data-path path/to/train.csv \
  --val-data-path path/to/val.csv \
  --test-data-path path/to/test.csv \
  --categorical-columns "col1,col2" \
  --numerical-columns "col3,col4" \
  --batch-size 128 \
  --num-epochs 10 \
  --learning-rate 0.001 \
  --output-dir ./outputs \
  --logger wandb \
  --wandb-project-name TabTransformer
```

Look at `train.py` to easily modify this for your needs

---


## 🤝 Contributing

I welcome contributions to enhance this project! Feel free to submit issues or pull requests.

---

## 📜 Citation

If you find this project useful, please cite it as:

```bibtex
@misc{tabtransformer-mlm,
    author = {},
    title = {TabTransformer with Masked Language Modeling: Unlocking Tabular Data Potential},
    year = {2024},
    howpublished = {\url{https://github.com/charitarthchugh/masked-tabtransformer-lightning}},
}
```

---

### Acknowledgements

I would like to thank Phil Wang (lucidrains) for his Tab-Transformer implementation. 