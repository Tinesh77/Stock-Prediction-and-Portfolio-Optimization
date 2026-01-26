def ensemble_prediction(predictions: dict, weights=None):
    """
    Weighted ensemble of model predictions.
    """
    if weights is None:
        weights = {
            "ARIMA": 0.3,
            "Random Forest": 0.4,
            "LSTM": 0.3
        }

    ensemble = 0.0
    for model, pred in predictions.items():
        ensemble += weights.get(model, 0) * pred

    return ensemble
