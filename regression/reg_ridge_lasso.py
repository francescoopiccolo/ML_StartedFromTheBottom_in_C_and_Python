import numpy as np

class LinearRegressionScratch:
    def __init__(self, lr=0.01, iters=1000, l1_pen=0, l2_pen=0):
        self.lr = lr
        self.iters = iters
        self.l1_pen = l1_pen  # Lambda per Lasso
        self.l2_pen = l2_pen  # Lambda per Ridge
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iters):
            # 1. Predizione lineare
            y_pred = np.dot(X, self.w) + self.b
            
            # 2. Calcolo dei gradienti base (MSE)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 3. AGGIUNTA PENALITÀ (Il cuore di Ridge e Lasso)
            # Se l2_pen > 0 -> Ridge
            dw += (self.l2_pen * 2 * self.w) / n_samples
            
            # Se l1_pen > 0 -> Lasso (usiamo np.sign per la derivata di |w|)
            dw += (self.l1_pen * np.sign(self.w)) / n_samples

            # 4. Aggiornamento parametri
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

# --- TEST CON DATI GIOCATTOLO ---
X_toy = np.array([[1], [2], [3], [4], [5]], dtype=float)
y_toy = np.array([2, 4, 5, 4, 5], dtype=float)

# Ridge: penalizza pesi grandi ma non li azzera
ridge_model = LinearRegressionScratch(l2_pen=10)
ridge_model.fit(X_toy, y_toy)

# Lasso: tende ad azzerare i pesi inutili
lasso_model = LinearRegressionScratch(l1_pen=10)
lasso_model.fit(X_toy, y_toy)

print(f"Pesi Ridge: {ridge_model.w}")
print(f"Pesi Lasso: {lasso_model.w}")