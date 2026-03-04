import numpy as np
import matplotlib.pyplot as plt

class KMeansScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.clusters = None

    def fit(self, X):
        # 1. Inizializzazione casuale dei centroidi
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # 2. Assegnazione: crea una lista di indici per ogni cluster
            self.clusters = [[] for _ in range(self.k)]
            for point_idx, x in enumerate(X):
                # Trova la distanza tra il punto x e tutti i centroidi
                distances = [np.sqrt(np.sum((x - c)**2)) for c in self.centroids]
                closest_idx = np.argmin(distances)
                self.clusters[closest_idx].append(point_idx)

            # Salva i vecchi centroidi per controllare la convergenza
            old_centroids = self.centroids.copy()

            # 3. Aggiornamento: ricalcola la media per ogni cluster
            for i in range(self.k):
                if self.clusters[i]: # Evita errori se un cluster è vuoto
                    self.centroids[i] = np.mean(X[self.clusters[i]], axis=0)

            # Se i centroidi non sono cambiati, abbiamo finito
            if np.all(old_centroids == self.centroids):
                break

    def predict(self, X):
        # Assegna nuovi punti ai cluster esistenti
        predictions = []
        for x in X:
            distances = [np.sqrt(np.sum((x - c)**2)) for c in self.centroids]
            predictions.append(np.argmin(distances))
        return np.array(predictions)

# --- Test con dati toy ---
X = np.random.randn(300, 2) + np.array([0, 0])
X = np.vstack([X, np.random.randn(300, 2) + np.array([5, 5])])
X = np.vstack([X, np.random.randn(300, 2) + np.array([0, 5])])

model = KMeansScratch(k=3)
model.fit(X)

# Visualizzazione
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='X', s=200)
plt.title("K-Means Clustering: I centroidi sono i punti rossi")
plt.show()