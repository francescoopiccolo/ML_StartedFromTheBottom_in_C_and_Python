#ifndef LINEAR_REG_H
#define LINEAR_REG_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double lr;          // Learning rate
    int iters;          // Numero iterazioni
    double l1_pen;      // Lambda per Lasso
    double l2_pen;      // Lambda per Ridge
    double *w;          // Pesi
    double b;           // Bias
    int n_features;
} LinearRegression;

LinearRegression* create_model(int n_features, double lr, int iters, double l1, double l2);
void fit(LinearRegression *model, double **X, double *y, int n_samples);
double predict(LinearRegression *model, double *x);
void free_model(LinearRegression *model);

#endif