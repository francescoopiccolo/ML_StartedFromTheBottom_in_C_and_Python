#include "naive_bayes.h"

int main() {
    int n_samples = 8;
    int n_features = 2;
    int n_classes = 2;

    // Allocazione dati training (Toy Data)
    double **X_train = (double**)malloc(n_samples * sizeof(double*));
    for(int i=0; i<n_samples; i++) X_train[i] = (double*)malloc(n_features * sizeof(double));
    
    int y_train[] = {0, 0, 0, 0, 1, 1, 1, 1};
    
    // Cani Piccoli
    X_train[0][0] = 20; X_train[0][1] = 5;
    X_train[1][0] = 25; X_train[1][1] = 7;
    X_train[2][0] = 18; X_train[2][1] = 4;
    X_train[3][0] = 22; X_train[3][1] = 6;
    // Cani Grandi
    X_train[4][0] = 60; X_train[4][1] = 30;
    X_train[5][0] = 65; X_train[5][1] = 35;
    X_train[6][0] = 70; X_train[6][1] = 40;
    X_train[7][0] = 55; X_train[7][1] = 28;

    NaiveBayes *nb = create_nb(n_classes, n_features);
    fit_nb(nb, X_train, y_train, n_samples);

    // Test su nuovi dati
    double test1[] = {21.0, 5.5}; // Ci aspettiamo 0
    double test2[] = {68.0, 38.0}; // Ci aspettiamo 1

    printf("Predizione test 1 (Piccolo): %d\n", predict_nb(nb, test1));
    printf("Predizione test 2 (Grande): %d\n", predict_nb(nb, test2));

    // Pulizia
    free_nb(nb);
    for(int i=0; i<n_samples; i++) free(X_train[i]);
    free(X_train);

    return 0;
}