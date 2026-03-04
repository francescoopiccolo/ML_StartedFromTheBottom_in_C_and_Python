#include <stdio.h>
#include <stdlib.h>
#include "adaboost.h"

int main() {
    int n_samples = 10;
    int n_features = 1;

    // Allocazione dati
    double **X = malloc(n_samples * sizeof(double*));
    int y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    for(int i=0; i<n_samples; i++) {
        X[i] = malloc(n_features * sizeof(double));
        X[i][0] = (double)(i * 2 + 1); // 1, 3, 5... 19
    }

    // Creazione e Training
    AdaBoost *model = create_adaboost(3);
    train_adaboost(model, X, y, n_samples, n_features);

    // Predizione
    printf("\n--- RISULTATI ADABOOST ---\n");
    for(int i = 0; i < n_samples; i++) {
        int p = predict_adaboost(model, X[i]);
        printf("Input: %4.1f | Pred: %d | Real: %d\n", X[i][0], p, y[i]);
    }

    // Pulizia
    for(int i=0; i<n_samples; i++) free(X[i]);
    free(X);
    free_adaboost(model);

    return 0;
}