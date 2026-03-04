import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr #learning rate
        self.iterations = iterations 
        self.weights = None 
        self.bias = None 

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Inizializzazione pesi: un peso per ogni feature
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.iterations):
            # Passaggio 1: Linear Model (z = Xw + b)
            linear_model = np.dot(X, self.weights) + self.bias
            # Passaggio 2: Applica Sigmoide (y_pred)
            y_predicted = self._sigmoid(linear_model)

            # Passaggio 3: Calcolo Gradienti (derivata della Log Loss)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Passaggio 4: Aggiornamento parametri
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in probs]
    

# Input (X): numeri da 1 a 20
X = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19]).reshape(-1, 1)

# Target (y): 0 se <= 10, 1 se > 10
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


# Inizializziamo il modello
model = LogisticRegressionScratch(lr=0.1, iterations=1000)

# Vediamo i pesi iniziali (casuali o zero)
print(f"Pesi iniziali: w={model.weights}, b={model.bias}")

model.fit(X, y)

print(f"Pesi finali: w={model.weights[0]:.4f}, b={model.bias:.4f}")

test_data = np.array([2, 18]).reshape(-1, 1)
predictions = model.predict(test_data)

for val, pred in zip(test_data.flatten(), predictions):
    print(f"Numero: {val} -> Classe Predetta: {pred} ({'Corretto!' if (val > 10) == pred else 'Sbagliato!'})")







def evaluate_classification(y_true, y_pred_probs, threshold=0.5):
    # Convertiamo le probabilità in classi 0 o 1 in base alla soglia
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Elementi della Confusion Matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calcolo Metriche
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "Confusion Matrix": [[tn, fp], [fn, tp]],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


def calculate_roc_auc(y_true, y_pred_probs):
    # Testiamo 100 soglie diverse da 1.0 a 0.0
    thresholds = np.linspace(1, 0, 100)
    tpr_list = [] # True Positive Rate
    fpr_list = [] # False Positive Rate
    
    for t in thresholds:
        y_p = (y_pred_probs >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_p == 1))
        fp = np.sum((y_true == 0) & (y_p == 1))
        fn = np.sum((y_true == 1) & (y_p == 0))
        tn = np.sum((y_true == 0) & (y_p == 0))
        
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    # Calcolo AUC usando la regola del trapezio (area sotto la curva)
    auc = np.trapz(tpr_list, fpr_list)
    
    return fpr_list, tpr_list, auc


# Dati reali: [Piccolo, Piccolo, Grande, Grande, Grande]
y_true = np.array([0, 0, 1, 1, 1])

# Probabilità predette dal modello (output della sigmoide)
y_probs = np.array([0.1, 0.4, 0.35, 0.8, 0.9]) 
# Nota: il terzo valore (0.35) è un errore del modello!

# 1. Metriche classiche
metrics = evaluate_classification(y_true, y_probs)
for k, v in metrics.items():
    print(f"{k}: {v}")

# 2. ROC e AUC
fpr, tpr, auc_value = calculate_roc_auc(y_true, y_probs)
print(f"AUC: {auc_value:.4f}")
