#ifndef SVM_H
#define SVM_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double *w;          // Pesi (vettore di dimensione n_features)
    double b;           // Bias (intercetta)
    double lr;          // Learning Rate
    double lambda;      // Parametro di regolarizzazione L2
    int n_iters;        // Numero di iterazioni
    int n_features;     // Numero di feature dei dati
} SVM;

SVM* create_svm(int n_features, double lr, double lambda, int n_iters);
void fit_svm(SVM *model, double **X, int *y, int n_samples);
int predict_svm(SVM *model, double *x);
void free_svm(SVM *model);

#endif