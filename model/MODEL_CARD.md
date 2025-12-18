
# Reentrancy Detection Model Card

## Model Information
- **Model Type**: Random Forest
- **Task**: Binary Classification (Reentrancy Detection)
- **Training Date**: 20251218_214550
- **Framework**: scikit-learn

## Performance Metrics (Test Set)
- **Accuracy**: 0.9682
- **Precision**: 0.9736
- **Recall**: 0.9625
- **F1-Score**: 0.9680
- **ROC-AUC**: 0.9959

## Training Data
- **Total Samples**: 37,576
- **Training Samples**: 30,060
- **Test Samples**: 7,516
- **Features**: 339 (Structural + TF-IDF)

## Model Details
- **Structural Features**: 39
- **TF-IDF Features**: 300
- **Total Features**: 339

## Usage
```python
import joblib

# Load model
model = joblib.load('reentrancy_model.pkl')
scaler = joblib.load('scaler.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Make prediction
prediction = model.predict(X_new)
```

## License
MIT License

## Contact
Opeyemi - Morgan State University
