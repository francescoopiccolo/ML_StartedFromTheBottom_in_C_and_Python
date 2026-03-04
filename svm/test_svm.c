#include "svm.h"

int main() {
    int n_samples = 6;
    int n_features = 2;

    // Allocazione X_train
    double **X_train = (double**)malloc(n_samples * sizeof(double*));
    for (int i = 0; i < n_samples; i++) X_train[i] = (double*)malloc(n_features * sizeof(double));

    // Dati toy (Classe -1 e Classe 1)
    X_train[0][0] = 1.0; X_train[0][1] = 2.0; // -1
    X_train[1][0] = 2.0; X_train[1][1] = 1.0; // -1
    X_train[2][0] = 2.0; X_train[2][1] = 3.0; // -1
    X_train[3][0] = 5.0; X_train[3][1] = 8.0; // 1
    X_train[4][0] = 6.0; X_train[4][1] = 7.0; // 1
    X_train[5][0] = 8.0; X_train[5][1] = 6.0; // 1

    int y_train[] = {-1, -1, -1, 1, 1, 1};

    // Creazione e addestramento
    SVM *svm = create_svm(n_features, 0.001, 0.01, 1000);
    fit_svm(svm, X_train, y_train, n_samples);

    // Test
    double test1[] = {1.0, 1.0}; // Dovrebbe essere -1
    double test2[] = {7.0, 7.0}; // Dovrebbe essere 1

    printf("Pesi: [%.4f, %.4f], Bias: %.4f\n", svm->w[0], svm->w[1], svm->b);
    printf("Predizione [1,1]: %d\n", predict_svm(svm, test1));
    printf("Predizione [7,7]: %d\n", predict_svm(svm, test2));

    // Pulizia
    free_svm(svm);
    for (int i = 0; i < n_samples; i++) free(X_train[i]);
    free(X_train);

    return 0;
}