import numpy as np

# --- CLASSE SVM DA SCRATCH ---
class SVMScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # SVM richiede classi -1 e 1
        y_transformed = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condizione di margine: y_i * (w·x_i - b) >= 1
                condition = y_transformed[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # Se il punto è fuori dal margine, riduciamo solo i pesi (regolarizzazione L2)
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Se il punto viola il margine, aggiorniamo w e b per correggere l'errore
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_transformed[idx]))
                    self.b -= self.lr * y_transformed[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# --- DATI TOY ---
# Rappresentano due gruppi distinti in uno spazio 2D
X_toy = np.array([
    [1, 2], [2, 1], [2, 3],   # Classe -1 (basso-sinistra)
    [5, 8], [6, 7], [8, 6]    # Classe 1  (alto-destra)
])
y_toy = np.array([-1, -1, -1, 1, 1, 1])

# --- ESECUZIONE ---
svm = SVMScratch(n_iters=1000)
svm.fit(X_toy, y_toy)

# Test su un nuovo punto
nuovo_punto = np.array([[4, 4], [1, 1]])
predizioni = svm.predict(nuovo_punto)

print(f"Pesi (w): {svm.w}")
print(f"Bias (b): {svm.b}")
print(f"Predizioni per [4,4] e [1,1]: {predizioni}")