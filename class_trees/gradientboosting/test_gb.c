#include <stdio.h>
#include <stdlib.h>
#include "gradient_boosting.h"

int main() {
    int n_samples = 10;
    int n_features = 1;

    double **X = malloc(n_samples * sizeof(double*));
    int y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    for(int i=0; i<n_samples; i++) {
        X[i] = malloc(n_features * sizeof(double));
        X[i][0] = (double)(i * 2 + 1);
    }

    // Configurazione: 10 estimatori, Learning Rate 0.1
    GradientBoosting *model = create_gb(10, 0.1);
    train_gb(model, X, y, n_samples, n_features);

    printf("\n--- RISULTATI GRADIENT BOOSTING ---\n");
    for(int i = 0; i < n_samples; i++) {
        int p = predict_gb(model, X[i]);
        printf("Input: %4.1f | Pred: %d | Real: %d\n", X[i][0], p, y[i]);
    }

    // Pulizia
    for(int i=0; i<n_samples; i++) free(X[i]);
    free(X);
    free_gb(model);

    return 0;
}