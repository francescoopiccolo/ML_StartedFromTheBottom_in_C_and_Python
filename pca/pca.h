#ifndef PCA_H
#define PCA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int n_components;
    int n_features;
    double *mean;
    double **components; // Matrice [n_components][n_features]
} PCA;

PCA* create_pca(int n_components, int n_features);
void fit_pca(PCA *model, double **X, int n_samples);
void transform_pca(PCA *model, double **X, double **X_proj, int n_samples);
void free_pca(PCA *model);

#endif