import mlflow.sklearn

class FraudScorer:
    def __init__(self, model_path, threshold):
        self.threshold = threshold
        self.model = mlflow.sklearn.load_model(model_path)
        
    def score(self, features):
        probability = float(self.model.predict_proba([features])[0][1])
        flagged = probability >= self.threshold
        return probability, flagged