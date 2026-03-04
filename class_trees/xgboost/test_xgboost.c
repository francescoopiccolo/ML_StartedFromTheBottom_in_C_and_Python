#include <stdio.h>
#include <stdlib.h>
#include "xgboost.h"

int main() {
    int n_samples = 10, n_features = 1;
    double **X = malloc(n_samples * sizeof(double*));
    int y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    for(int i=0; i<n_samples; i++) {
        X[i] = malloc(n_features * sizeof(double));
        X[i][0] = (double)(i * 2 + 1);
    }

    XGBoost *model = create_xgb(5, 0.1, 1.0);
    train_xgb(model, X, y, n_samples, n_features);

    printf("\n--- RISULTATI XGBOOST ---\n");
    for(int i = 0; i < n_samples; i++) {
        double prob = predict_proba_xgb(model, X[i]);
        int pred = predict_xgb(model, X[i]);
        printf("X: %4.1f | Prob: %.4f | Pred: %d | Real: %d\n", X[i][0], prob, pred, y[i]);
    }

    // Cleanup...
    return 0;
}