import numpy as np
from collections import Counter

class KNNScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # Il k-NN non "impara", memorizza solo i dati
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # 1. Calcolo la distanza Euclidea tra x e tutti i punti nel train set
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        
        # 2. Ottengo gli indici dei k punti più vicini (ordinamento)
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Estraggo le etichette di questi k punti
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Maggioranza dei voti (Majority Vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# --- DATI GIOCATTOLO ---
X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 7], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1]) # 0: Basso, 1: Alto

knn = KNNScratch(k=3)
knn.fit(X_train, y_train)

# Test su un nuovo punto
X_new = np.array([[4, 4]])
pred = knn.predict(X_new)
print(f"Predizione per [4, 4]: {pred[0]}")