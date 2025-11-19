import os
import joblib
import numpy as np

try:
    from src.config import MODELS_DIR
except ModuleNotFoundError:
    from config import MODELS_DIR


class FraudEnsemble:
    def __init__(
        self,
        binary_threshold: float = 0.40,
        safe_threshold: float = 0.05,
        fraud_threshold: float = 0.40,
    ):
        self.binary_threshold = binary_threshold
        self.safe_threshold = safe_threshold
        self.fraud_threshold = fraud_threshold

        self.models = {}
        self.scaler = None

        self._load_components()

    # -------------------------------------------------------
    def _load_components(self):
        model_files = {
            "logreg": "logreg_model.pkl",
            "naive_bayes": "naive_bayes_model.pkl",
            "random_forest": "random_forest_model.pkl",
            "lightgbm": "lightgbm_model.pkl",
        }

        for name, filename in model_files.items():
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"Loaded {name} model.")
            else:
                print(f"WARNING: missing {name} model → skipping.")

        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Missing scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")

    # -------------------------------------------------------
    def _raw_ensemble_proba(self, X):
        """
        Return raw (uncalibrated) ensemble probability array for input X.
        X may be:
          - 1D array-like of length n_features -> treated as single sample
          - 2D array-like (n_samples, n_features)
        Returns numpy array shape (n_samples,)
        """
        X = np.array(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = self.scaler.transform(X)

        probs = []
        for name, model in self.models.items():
            try:
                p = model.predict_proba(X)[:, 1]
                probs.append(p)
            except Exception as e:
                print(f"Model {name} predict_proba failed: {e}")

        if len(probs) == 0:
            raise RuntimeError("No models available in ensemble.")

        return np.mean(np.vstack(probs), axis=0)

    # -------------------------------------------------------
    def _boost_probability(self, p: np.ndarray):
        """
        Gentle boost to make probabilities more visually intuitive.
        Works elementwise on numpy arrays.
        """
        p = np.array(p, dtype=float)
        boost_strength = 0.35
        boosted = p + boost_strength * (p - 0.5)
        boosted = np.clip(boosted, 0.0, 1.0)
        return boosted

    # -------------------------------------------------------
    def predict_proba(self, X):
        """
        Accepts single sample (list/1d) or batch (2d array).
        Returns scalar float for single sample, numpy array for batch.
        """
        raw = self._raw_ensemble_proba(X)
        boosted = self._boost_probability(raw)

        # Return scalar if single sample input
        if np.array(X).ndim == 1 or (hasattr(X, "shape") and getattr(X, "shape")[0] == 1):
            return float(boosted[0])
        return boosted

    # -------------------------------------------------------
    def predict(self, X):
        p = self.predict_proba(X)
        return int(p >= self.binary_threshold) if np.isscalar(p) else (p >= self.binary_threshold).astype(int)

    # -------------------------------------------------------
    def predict_risk_label(self, X):
        p = float(self.predict_proba(X))

        if p < self.safe_threshold:
            return "SAFE TRANSACTION"
        if p > self.fraud_threshold:
            return "FRAUD DETECTED"
        return "REVIEW REQUIRED"