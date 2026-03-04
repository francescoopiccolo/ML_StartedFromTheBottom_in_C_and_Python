#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "xgboost.h"

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

XGBoost* create_xgb(int n_estimators, double lr, double lambd) {
    XGBoost *model = malloc(sizeof(XGBoost));
    model->n_estimators = n_estimators;
    model->lr = lr;
    model->lambd = lambd;
    model->init_log_odds = 0.0; // Corrisponde a probabilità 0.5
    model->trees = malloc(n_estimators * sizeof(XGBoostTree));
    return model;
}

void train_xgb(XGBoost *model, double **X, int *y, int n_samples, int n_features) {
    double *log_odds = malloc(n_samples * sizeof(double));
    double *g = malloc(n_samples * sizeof(double)); // Gradiente
    double *h = malloc(n_samples * sizeof(double)); // Hessiano

    for (int i = 0; i < n_samples; i++) log_odds[i] = model->init_log_odds;

    for (int t = 0; t < model->n_estimators; t++) {
        // 1. Calcolo gradienti e hessiani per LogLoss
        for (int i = 0; i < n_samples; i++) {
            double p = sigmoid(log_odds[i]);
            g[i] = p - y[i];
            h[i] = p * (1.0 - p);
        }

        XGBoostTree best_tree;
        double max_gain = 0;

        // Similarity Score della radice
        double sum_g_total = 0, sum_h_total = 0;
        for(int i=0; i<n_samples; i++) { sum_g_total += g[i]; sum_h_total += h[i]; }
        double root_sim = (sum_g_total * sum_g_total) / (sum_h_total + model->lambd);

        // 2. Ricerca dello split che massimizza il Gain
        for (int f = 0; f < n_features; f++) {
            for (int i = 0; i < n_samples; i++) {
                double threshold = X[i][f];
                double g_l = 0, h_l = 0, g_r = 0, h_r = 0;
                int count_l = 0, count_r = 0;

                for (int j = 0; j < n_samples; j++) {
                    if (X[j][f] <= threshold) { g_l += g[j]; h_l += h[j]; count_l++; }
                    else { g_r += g[j]; h_r += h[j]; count_r++; }
                }

                if (count_l == 0 || count_r == 0) continue;

                double sim_l = (g_l * g_l) / (h_l + model->lambd);
                double sim_r = (g_r * g_r) / (h_r + model->lambd);
                double gain = sim_l + sim_r - root_sim;

                if (gain > max_gain) {
                    max_gain = gain;
                    best_tree.feature_idx = f;
                    best_tree.threshold = threshold;
                    // Output ottimale delle foglie: -Sum(g) / (Sum(h) + lambda)
                    best_tree.left_val = -g_l / (h_l + model->lambd);
                    best_tree.right_val = -g_r / (h_r + model->lambd);
                }
            }
        }

        // 3. Aggiornamento Log-Odds
        for (int i = 0; i < n_samples; i++) {
            double out = (X[i][best_tree.feature_idx] <= best_tree.threshold) ? best_tree.left_val : best_tree.right_val;
            log_odds[i] += model->lr * out;
        }

        model->trees[t] = best_tree;
        printf("Iterazione %d: Max Gain = %.4f\n", t + 1, max_gain);
    }

    free(log_odds); free(g); free(h);
}

double predict_proba_xgb(XGBoost *model, double *x) {
    double log_odds = model->init_log_odds;
    for (int i = 0; i < model->n_estimators; i++) {
        XGBoostTree t = model->trees[i];
        log_odds += model->lr * ((x[t.feature_idx] <= t.threshold) ? t.left_val : t.right_val);
    }
    return sigmoid(log_odds);
}

int predict_xgb(XGBoost *model, double *x) {
    return predict_proba_xgb(model, x) >= 0.5 ? 1 : 0;
}

void free_xgb(XGBoost *model) {
    free(model->trees);
    free(model);
}