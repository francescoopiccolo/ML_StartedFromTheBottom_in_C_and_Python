#ifndef KNN_H
#define KNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double **X_train;
    int *y_train;
    int n_samples;
    int n_features;
    int k;
} KNN;

// Struttura di supporto per l'ordinamento delle distanze
typedef struct {
    double distance;
    int label;
} Neighbor;

KNN* create_knn(int k);
void fit_knn(KNN *model, double **X, int *y, int n_samples, int n_features);
int predict_knn(KNN *model, double *x);
void free_knn(KNN *model);

#endif