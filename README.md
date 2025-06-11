# 🌍 Multilingual Language Detection using XLM-RoBERTa

This project aims to build a multilingual text classifier that detects the **language of a given sentence**. Leveraging the power of **XLM-RoBERTa**, an open-source transformer model trained on 100 languages, this solution provides robust and accurate results across diverse linguistic inputs.

## 🔓 Open Source Foundation

This project uses [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base), an open-source model developed by **Meta AI** and available under the **MIT License**. It is integrated using Hugging Face's `transformers` library.

---

## 🏗️ Project Pipeline Overview

1. **Data Loading**:  
   Load the multilingual dataset (`Language Detection.csv`) using Polars for speed and efficiency.

2. **Class Imbalance Handling**:  
   Analyze language class distributions and balance the dataset using additional sentences from `.tsv` and `.txt` corpora stored in `extra datasets/`.

3. **Preprocessing**:  
   - Encode labels using `LabelEncoder`
   - Tokenize sentences using `AutoTokenizer` for `xlm-roberta-base`
   - Truncate and pad sequences for uniform length

4. **Model Definition**:  
   Use `TFAutoModelForSequenceClassification` with:
   - 1 hidden classification layer
   - Sparse categorical crossentropy loss
   - Adam optimizer

5. **Training**:  
   - Early stopping to prevent overfitting
   - Model checkpointing to save best weights
   - Learning rate reduction on plateau
   - Saving the training logs

6. **Evaluation**:  
   - Plot training/validation accuracy & loss
   - Predict using TensorFlow's build-in ```model.evaluate()``` syntax
   - Generate a detailed classification report
   - (Optional) Confusion matrix per class

---

## 📊 Dataset

- **Base CSV**: `Language Detection.csv` (text + label)
- **Additions** (to remove class imbalance):  
  - `.tsv` files per language
  - `.txt` files with sentence-level extraction logic

---

## 🧠 Model Configuration

| Component        | Description                                       |
|------------------|---------------------------------------------------|
| Model            | `xlm-roberta-base`                                |
| Layers           | 12-layer Transformer + Dense Classification Head  |
| Tokenizer        | HuggingFace `AutoTokenizer`                       |
| Max Length       | 128 tokens                                        |
| Loss             | `SparseCategoricalCrossentropy(from_logits=True)` |
| Optimizer        | Adam (LR: 1e-5)                                   |
| Metrics          | Accuracy, F1-score, Classification Report         |

---

## 💡 Requirements

See [`requirements.txt`](./requirements.txt)

---

## 💻 Running Instructions

1. **Clone and Install**:

    ```bash
    git clone https://github.com/Avik816/multilinguinal-language-detection.git
    cd multilinguinal-language-detection
    pip install -r requirements.txt
    ```

2. **Prepare Data**:
   - Place `Language Detection.csv` and `extra datasets/*.tsv` or `*.txt` files in the root or appropriate folder.

3. **Run Notebook**:
   ```bash
   jupyter notebook "language detection.ipynb"
   ```

---

## 📁 Project Structure

```
language-detection-transformers/
├── language detection.ipynb       # Main Jupyter Notebook for training and evaluation
├── requirements.txt               # List of dependencies and versions (for pip install)
├── README.md                      # Project overview, instructions, and architecture
├── Language Detection.csv         # Primary dataset with 'Text' and 'Language' columns
├── extra datasets/                # Additional multilingual datasets for augmentation
├── models/                  # Checkpoints and fully saved fine-tuned models
└── logs/                          # To save the training logs of the model during fine-tuning

```

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).  
It uses the open-source [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base) model, released by Meta AI under the MIT license and distributed via [Hugging Face Transformers](https://huggingface.co/docs/transformers).  
You are free to use, modify, and distribute this project under the terms of the license.
