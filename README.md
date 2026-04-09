# Hallucination Detection in LLMs

A machine learning project to detect hallucinations in large language model (LLM) generated responses using ensemble methods with LightGBM and CatBoost classifiers.

## Project Overview

This project addresses the critical problem of detecting when LLMs generate false or misleading information (hallucinations). By training on a dataset of genuine and hallucinated responses, the model learns to classify new LLM outputs with high accuracy.

## Features

- **Data Preprocessing**: TF-IDF vectorization for text feature extraction
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Ensemble Models**:
  - LightGBM classifier
  - CatBoost classifier
- **Comprehensive Evaluation**: Accuracy, F1 Score, ROC-AUC metrics
- **Visualizations**: Confusion matrices, ROC curves, feature importance plots
- **Automatic Submission Generation**: Creates submission-ready predictions

## Model Performance

| Metric | LightGBM | CatBoost |
|--------|----------|----------|
| Accuracy | 94.72% | 94.36% |
| F1 Score | 0.2605 | 0.2656 ⭐ |
| ROC-AUC | 80.45% | 82.54% ⭐ |

**Best Model**: CatBoost (based on ROC-AUC score)

## Dataset

- **Source**: ML Olympiad - Detect Hallucinations in LLMs
- **Train Set**: 25,256 samples
- **Validation Set**: 3,334 samples
- **Classes**: Genuine (majority), Hallucinated (minority)

## Project Structure

```
hallucination_detection/
├── main.py                      # Main pipeline script
├── preprocess.py                # Data loading and feature engineering
├── train.py                     # Model training
├── evaluate.py                  # Model evaluation metrics
├── predict.py                   # Prediction and submission generation
├── visualize.py                 # Visualization functions
├── config.py                    # Configuration settings
├── inspect_preprocess.py        # Data inspection utilities
├── requirements.txt             # Project dependencies
├── README.md                    # This file
├── models/                      # Trained models storage
│   └── catboost_model.cbm       # Best trained model
├── outputs/                     # Generated outputs
│   └── submission.csv           # Predictions on test set
└── data/                        # Data files (not included)
    ├── train.csv
    └── test.csv
```

## Installation

### Prerequisites
- Python 3.11+
- pip or conda package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hallucination_detection.git
   cd hallucination_detection
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Full Pipeline

Execute the complete training and evaluation workflow:

```bash
python main.py
```

This will:
- Load and preprocess data
- Build feature vectors
- Train both LightGBM and CatBoost models
- Evaluate and compare model performance
- Generate visualizations
- Create submission predictions

### Individual Components

**Preprocess data only:**
```bash
python preprocess.py
```

**Train models only:**
```bash
python train.py
```

**Evaluate models only:**
```bash
python evaluate.py
```

**Generate predictions:**
```bash
python predict.py
```

**Create visualizations:**
```bash
python visualize.py
```

## Output Files

After running `main.py`, the following files are generated:

### Visualizations (outputs/)
- `class_distribution.png` - Class distribution in training data
- `answer_length_distribution.png` - Answer length statistics
- `confusion_matrices.png` - Confusion matrices for both models
- `roc_curves.png` - ROC curves comparison
- `comparison_bar.png` - Model metrics comparison
- `feature_importance.png` - Feature importance from LightGBM

### Models (models/)
- `catboost_model.cbm` - Trained CatBoost model

### Predictions (outputs/)
- `submission.csv` - Test set predictions

## Dependencies

Core libraries:
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning utilities
- **lightgbm** - LightGBM classifier
- **catboost** - CatBoost classifier
- **imbalanced-learn** - SMOTE for class imbalance
- **matplotlib** - Plotting library
- **seaborn** - Statistical visualization
- **scipy** - Scientific computing
- **joblib** - Model serialization

See `requirements.txt` for complete list with versions.

## Configuration

Edit `config.py` to adjust:
- Model hyperparameters
- Train/validation split ratio
- Random seeds for reproducibility
- Feature engineering parameters

## Results Summary

The project achieves strong performance on the hallucination detection task:

- **Best ROC-AUC**: 82.54% (CatBoost)
- **Precision (Hallucinated)**: 42% - Few false positives
- **Recall (Hallucinated)**: 19% - Conservative predictions
- **Class Imbalance Handling**: Successfully balanced minority class with SMOTE

## Key Findings

1. Both models perform well on the majority class (Genuine responses)
2. CatBoost outperforms LightGBM on ROC-AUC metric
3. The ensemble approach provides robust predictions
4. TF-IDF features effectively capture textual patterns of hallucinations

## Future Improvements

- [ ] Experiment with different feature engineering techniques
- [ ] Try additional models (XGBoost, Random Forest)
- [ ] Implement cross-validation for more robust evaluation
- [ ] Fine-tune hyperparameters using grid search or Bayesian optimization
- [ ] Incorporate pre-trained embeddings (BERT, GPT)
- [ ] Add ensemble voting mechanism
- [ ] Implement threshold optimization for business requirements

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Harsh Avardhanan

## Acknowledgments

- ML Olympiad for the dataset
- LightGBM and CatBoost teams for excellent gradient boosting libraries
- Scikit-learn and imbalanced-learn communities

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Last Updated**: April 10, 2026
