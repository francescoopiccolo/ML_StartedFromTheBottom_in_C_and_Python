#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "random_forest.h"

int main() {
    srand(time(NULL));

    // Dataset giocattolo
    int n_samples = 10, n_features = 1;
    double **X = malloc(n_samples * sizeof(double*));
    int y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    for(int i=0; i<n_samples; i++) {
        X[i] = malloc(n_features * sizeof(double));
        X[i][0] = (double)(i * 2 + 1);
    }

    // Configurazione: 10 alberi, prof 3, usa 1 feature per split
    RandomForest *rf = create_rf(10, 3, 1);
    train_rf(rf, X, y, n_samples, n_features);

    printf("Predizioni su tutto il dataset:\n");
    for(int i=0; i<n_samples; i++) {
        printf("X: %.1f -> Pred: %d\n", X[i][0], predict_rf(rf, X[i]));
    }

    return 0;
}