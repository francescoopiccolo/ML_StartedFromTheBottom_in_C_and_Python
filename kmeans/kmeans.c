#include "kmeans.h"

KMeans* create_kmeans(int k, int n_features, int max_iters) {
    KMeans *model = (KMeans*)malloc(sizeof(KMeans));
    model->k = k;
    model->n_features = n_features;
    model->max_iters = max_iters;
    model->centroids = (double**)malloc(k * sizeof(double*));
    for (int i = 0; i < k; i++) {
        model->centroids[i] = (double*)calloc(n_features, sizeof(double));
    }
    return model;
}

double dist_sq(double *p1, double *p2, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += pow(p1[i] - p2[i], 2);
    return sum;
}

void fit_kmeans(KMeans *model, double **X, int n_samples) {
    srand(time(NULL));
    int nf = model->n_features;
    int k = model->k;

    // 1. Inizializzazione casuale dei centroidi
    for (int i = 0; i < k; i++) {
        int r = rand() % n_samples;
        for (int j = 0; j < nf; j++) model->centroids[i][j] = X[r][j];
    }

    int *assignments = (int*)malloc(n_samples * sizeof(int));

    for (int iter = 0; iter < model->max_iters; iter++) {
        bool changed = false;

        // 2. Assegnazione
        for (int i = 0; i < n_samples; i++) {
            int best_c = 0;
            double min_d = dist_sq(X[i], model->centroids[0], nf);
            for (int c = 1; c < k; c++) {
                double d = dist_sq(X[i], model->centroids[c], nf);
                if (d < min_d) {
                    min_d = d;
                    best_c = c;
                }
            }
            if (assignments[i] != best_c) {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if (!changed && iter > 0) break;

        // 3. Aggiornamento (Media dei punti)
        double **new_centroids = (double**)malloc(k * sizeof(double*));
        int *counts = (int*)calloc(k, sizeof(int));
        for (int i = 0; i < k; i++) new_centroids[i] = (double*)calloc(nf, sizeof(double));

        for (int i = 0; i < n_samples; i++) {
            int c = assignments[i];
            counts[c]++;
            for (int j = 0; j < nf; j++) new_centroids[c][j] += X[i][j];
        }

        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                for (int j = 0; j < nf; j++) model->centroids[i][j] = new_centroids[i][j] / counts[i];
            }
            free(new_centroids[i]);
        }
        free(new_centroids);
        free(counts);
    }
    free(assignments);
}

int predict_kmeans(KMeans *model, double *x) {
    int best_c = 0;
    double min_d = dist_sq(x, model->centroids[0], model->n_features);
    for (int c = 1; c < model->k; c++) {
        double d = dist_sq(x, model->centroids[c], model->n_features);
        if (d < min_d) {
            min_d = d;
            best_c = c;
        }
    }
    return best_c;
}

void free_kmeans(KMeans *model) {
    for (int i = 0; i < model->k; i++) free(model->centroids[i]);
    free(model->centroids);
    free(model);
}