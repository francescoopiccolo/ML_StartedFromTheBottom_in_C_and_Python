#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gradient_boosting.h"

GradientBoosting* create_gb(int n_estimators, double lr) {
    GradientBoosting *model = malloc(sizeof(GradientBoosting));
    model->n_estimators = n_estimators;
    model->lr = lr;
    model->trees = malloc(n_estimators * sizeof(RegressionStump));
    return model;
}

void train_gb(GradientBoosting *model, double **X, int *y, int n_samples, int n_features) {
    // 1. Calcolo predizione iniziale (media di y)
    double sum_y = 0;
    for (int i = 0; i < n_samples; i++) sum_y += y[i];
    model->init_pred = sum_y / n_samples;

    double *current_preds = malloc(n_samples * sizeof(double));
    double *residuals = malloc(n_samples * sizeof(double));
    for (int i = 0; i < n_samples; i++) current_preds[i] = model->init_pred;

    for (int t = 0; t < model->n_estimators; t++) {
        // 2. Calcolo dei residui: r = y - f(x)
        for (int i = 0; i < n_samples; i++) residuals[i] = y[i] - current_preds[i];

        RegressionStump best_stump;
        double min_mse = 1e30;

        // 3. Fitting del Regression Stump sui residui
        for (int f = 0; f < n_features; f++) {
            for (int i = 0; i < n_samples; i++) {
                double threshold = X[i][f];
                
                int n_l = 0, n_r = 0;
                double sum_l = 0, sum_r = 0;
                
                for (int j = 0; j < n_samples; j++) {
                    if (X[j][f] <= threshold) { n_l++; sum_l += residuals[j]; }
                    else { n_r++; sum_r += residuals[j]; }
                }

                if (n_l == 0 || n_r == 0) continue;

                double l_val = sum_l / n_l;
                double r_val = sum_r / n_r;

                // Calcolo MSE (Errore Quadratico Medio)
                double mse = 0;
                for (int j = 0; j < n_samples; j++) {
                    double p = (X[j][f] <= threshold) ? l_val : r_val;
                    mse += pow(residuals[j] - p, 2);
                }

                if (mse < min_mse) {
                    min_mse = mse;
                    best_stump.feature_idx = f;
                    best_stump.threshold = threshold;
                    best_stump.left_val = l_val;
                    best_stump.right_val = r_val;
                }
            }
        }

        // 4. Aggiornamento predizioni: f(x) = f(x) + lr * tree(x)
        for (int i = 0; i < n_samples; i++) {
            double p = (X[i][best_stump.feature_idx] <= best_stump.threshold) ? best_stump.left_val : best_stump.right_val;
            current_preds[i] += model->lr * p;
        }

        model->trees[t] = best_stump;
        printf("Albero %d training... MSE Residuo: %.4f\n", t + 1, min_mse / n_samples);
    }

    free(current_preds);
    free(residuals);
}

int predict_gb(GradientBoosting *model, double *x) {
    double y_pred = model->init_pred;
    for (int i = 0; i < model->n_estimators; i++) {
        RegressionStump s = model->trees[i];
        double val = (x[s.feature_idx] <= s.threshold) ? s.left_val : s.right_val;
        y_pred += model->lr * val;
    }
    // Per classificazione: soglia a 0.5
    return (y_pred >= 0.5) ? 1 : 0;
}

void free_gb(GradientBoosting *model) {
    free(model->trees);
    free(model);
}