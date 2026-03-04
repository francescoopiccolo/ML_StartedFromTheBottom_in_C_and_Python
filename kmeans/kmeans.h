#ifndef KMEANS_H
#define KMEANS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

typedef struct {
    int k;
    int max_iters;
    int n_features;
    double **centroids;
} KMeans;

KMeans* create_kmeans(int k, int n_features, int max_iters);
void fit_kmeans(KMeans *model, double **X, int n_samples);
int predict_kmeans(KMeans *model, double *x);
void free_kmeans(KMeans *model);

#endif