#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "adaboost.h"

AdaBoost* create_adaboost(int n_stumps) {
    AdaBoost *model = malloc(sizeof(AdaBoost));
    model->n_stumps = n_stumps;
    model->stumps = malloc(n_stumps * sizeof(DecisionStump));
    return model;
}

void train_adaboost(AdaBoost *model, double **X, int *y, int n_samples, int n_features) {
    // 1. Inizializzazione pesi w = 1/N
    double *w = malloc(n_samples * sizeof(double));
    for (int i = 0; i < n_samples; i++) w[i] = 1.0 / n_samples;

    for (int s = 0; s < model->n_stumps; s++) {
        double min_error = 1e30; // Infinito
        DecisionStump best_stump;

        // 2. Ricerca del miglior Stump (Brute force su feature, soglie e polarità)
        for (int f = 0; f < n_features; f++) {
            for (int i = 0; i < n_samples; i++) {
                double threshold = X[i][f];
                int polarities[] = {1, -1};

                for (int p = 0; p < 2; p++) {
                    int current_polarity = polarities[p];
                    double error = 0;

                    // Calcolo errore pesato
                    for (int j = 0; j < n_samples; j++) {
                        int prediction = (current_polarity * X[j][f] <= current_polarity * threshold) ? 0 : 1;
                        if (prediction != y[j]) {
                            error += w[j];
                        }
                    }

                    if (error < min_error) {
                        min_error = error;
                        best_stump.feature_idx = f;
                        best_stump.threshold = threshold;
                        best_stump.polarity = current_polarity;
                    }
                }
            }
        }

        // 3. Calcolo Alpha (affidabilità dello stump)
        double eps = 1e-10;
        best_stump.alpha = 0.5 * log((1.0 - min_error + eps) / (min_error + eps));

        // 4. Aggiornamento pesi dei campioni
        double sum_w = 0;
        for (int i = 0; i < n_samples; i++) {
            int pred = (best_stump.polarity * X[i][best_stump.feature_idx] <= best_stump.polarity * best_stump.threshold) ? 0 : 1;
            
            // Convertiamo 0/1 in -1/1 per la formula: w = w * exp(-alpha * y * p)
            int y_math = (y[i] == 0) ? -1 : 1;
            int p_math = (pred == 0) ? -1 : 1;
            
            w[i] *= exp(-best_stump.alpha * y_math * p_math);
            sum_w += w[i];
        }

        // Normalizzazione pesi
        for (int i = 0; i < n_samples; i++) w[i] /= sum_w;

        model->stumps[s] = best_stump;
        printf("Stump %d addestrato. Errore: %.4f, Alpha: %.4f\n", s+1, min_error, best_stump.alpha);
    }
    free(w);
}

int predict_adaboost(AdaBoost *model, double *x) {
    double final_score = 0;
    for (int i = 0; i < model->n_stumps; i++) {
        DecisionStump s = model->stumps[i];
        int pred = (s.polarity * x[s.feature_idx] <= s.polarity * s.threshold) ? 0 : 1;
        int p_math = (pred == 0) ? -1 : 1;
        final_score += s.alpha * p_math;
    }
    return (final_score >= 0) ? 1 : 0;
}

void free_adaboost(AdaBoost *model) {
    free(model->stumps);
    free(model);
}