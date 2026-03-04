#include "knn.h"

KNN* create_knn(int k) {
    KNN *model = (KNN*)malloc(sizeof(KNN));
    model->k = k;
    return model;
}

void fit_knn(KNN *model, double **X, int *y, int n_samples, int n_features) {
    model->X_train = X;
    model->y_train = y;
    model->n_samples = n_samples;
    model->n_features = n_features;
}

// Funzione di comparazione per qsort
int compare_neighbors(const void *a, const void *b) {
    Neighbor *n1 = (Neighbor *)a;
    Neighbor *n2 = (Neighbor *)b;
    if (n1->distance < n2->distance) return -1;
    if (n1->distance > n2->distance) return 1;
    return 0;
}

double euclidean_distance(double *v1, double *v2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += pow(v1[i] - v2[i], 2);
    }
    return sqrt(sum);
}

int predict_knn(KNN *model, double *x) {
    Neighbor *neighbors = (Neighbor*)malloc(model->n_samples * sizeof(Neighbor));

    // 1. Calcolo tutte le distanze
    for (int i = 0; i < model->n_samples; i++) {
        neighbors[i].distance = euclidean_distance(x, model->X_train[i], model->n_features);
        neighbors[i].label = model->y_train[i];
    }

    // 2. Ordinamento per distanza (Equivalente a np.argsort)
    qsort(neighbors, model->n_samples, sizeof(Neighbor), compare_neighbors);

    // 3. Votazione (Majority Vote sui primi K)
    // Assumiamo etichette semplici (es. 0 e 1). Per generalizzare useremmo una hashmap o un array di conteggio
    int max_label = 0;
    for(int i=0; i < model->n_samples; i++) if(model->y_train[i] > max_label) max_label = model->y_train[i];
    
    int *counts = (int*)calloc(max_label + 1, sizeof(int));
    for (int i = 0; i < model->k; i++) {
        counts[neighbors[i].label]++;
    }

    int result_label = 0;
    int max_votes = -1;
    for (int i = 0; i <= max_label; i++) {
        if (counts[i] > max_votes) {
            max_votes = counts[i];
            result_label = i;
        }
    }

    free(neighbors);
    free(counts);
    return result_label;
}

void free_knn(KNN *model) {
    free(model);
}