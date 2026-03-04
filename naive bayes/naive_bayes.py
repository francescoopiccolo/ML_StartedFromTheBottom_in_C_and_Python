import numpy as np

# --- CODICE NAIVE BAYES (La tua classe) ---
class NaiveBayesScratch:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return np.array([self.__predict(x) for x in X])

    def __predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            # Aggiungiamo un piccolo epsilon per evitare log(0) se la PDF è zero
            posterior = np.sum(np.log(self._pdf(idx, x) + 1e-10))
            posteriors.append(prior + posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# --- DATI GIOCATTOLO (Toy Data) ---
# Feature 1: Altezza (cm), Feature 2: Peso (kg)
# Classe 0: Cani di piccola taglia, Classe 1: Cani di grande taglia
X_train = np.array([
    [20, 5], [25, 7], [18, 4], [22, 6],  # Classe 0
    [60, 30], [65, 35], [70, 40], [55, 28] # Classe 1
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# --- ESECUZIONE ---
nb = NaiveBayesScratch()
nb.fit(X_train, y_train)

# Test su nuovi dati "sconosciuti"
X_test = np.array([
    [21, 5.5],  # Dovrebbe essere 0
    [68, 38]    # Dovrebbe essere 1
])

predizioni = nb.predict(X_test)

print("Statistiche calcolate (Medie per classe):")
print(nb._mean)
print(f"\nPredizioni per i nuovi dati: {predizioni}")