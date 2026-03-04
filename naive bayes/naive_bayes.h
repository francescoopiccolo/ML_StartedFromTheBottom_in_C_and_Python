#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int n_classes;
    int n_features;
    double **means;   // Matrice [n_classes][n_features]
    double **vars;    // Matrice [n_classes][n_features]
    double *priors;   // Array [n_classes]
    int *classes;     // Label delle classi
} NaiveBayes;

// Funzioni principali
NaiveBayes* create_nb(int n_classes, int n_features);
void fit_nb(NaiveBayes *model, double **X, int *y, int n_samples);
int predict_nb(NaiveBayes *model, double *x);
void free_nb(NaiveBayes *model);

#endif