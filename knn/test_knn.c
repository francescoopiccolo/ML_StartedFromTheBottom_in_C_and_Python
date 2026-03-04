#include "knn.h"

int main() {
    int n_samples = 6;
    int n_features = 2;
    int k = 3;

    // Allocazione dati training
    double **X_train = (double**)malloc(n_samples * sizeof(double*));
    for (int i = 0; i < n_samples; i++) X_train[i] = (double*)malloc(n_features * sizeof(double));

    // Dati: [1,2], [2,3], [3,3] -> Classe 0 | [6,5], [7,7], [8,6] -> Classe 1
    X_train[0][0] = 1.0; X_train[0][1] = 2.0;
    X_train[1][0] = 2.0; X_train[1][1] = 3.0;
    X_train[2][0] = 3.0; X_train[2][1] = 3.0;
    X_train[3][0] = 6.0; X_train[3][1] = 5.0;
    X_train[4][0] = 7.0; X_train[4][1] = 7.0;
    X_train[5][0] = 8.0; X_train[5][1] = 6.0;

    int y_train[] = {0, 0, 0, 1, 1, 1};

    KNN *knn = create_knn(k);
    fit_knn(knn, X_train, y_train, n_samples, n_features);

    // Test su nuovo punto [4, 4]
    double test_point[] = {4.0, 4.0};
    int prediction = predict_knn(knn, test_point);

    printf("Predizione per il punto [4.0, 4.0]: %d\n", prediction);

    // Pulizia
    for (int i = 0; i < n_samples; i++) free(X_train[i]);
    free(X_train);
    free_knn(knn);

    return 0;
}