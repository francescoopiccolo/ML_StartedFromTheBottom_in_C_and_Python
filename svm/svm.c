#include "svm.h"

SVM* create_svm(int n_features, double lr, double lambda, int n_iters) {
    SVM *model = (SVM*)malloc(sizeof(SVM));
    model->n_features = n_features;
    model->lr = lr;
    model->lambda = lambda;
    model->n_iters = n_iters;
    model->w = (double*)calloc(n_features, sizeof(double)); // Inizializza a zero
    model->b = 0.0;
    return model;
}

double dot_product(double *v1, double *v2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += v1[i] * v2[i];
    return sum;
}

void fit_svm(SVM *model, double **X, int *y, int n_samples) {
    for (int iter = 0; iter < model->n_iters; iter++) {
        for (int i = 0; i < n_samples; i++) {
            // Calcolo y_i * (w · x_i - b)
            double condition = y[i] * (dot_product(X[i], model->w, model->n_features) - model->b);

            if (condition >= 1) {
                // Punto correttamente classificato fuori dal margine
                // dw = 2 * lambda * w
                for (int f = 0; f < model->n_features; f++) {
                    model->w[f] -= model->lr * (2 * model->lambda * model->w[f]);
                }
            } else {
                // Il punto viola il margine (Hinge Loss > 0)
                // dw = 2 * lambda * w - y_i * x_i
                // db = y_i
                for (int f = 0; f < model->n_features; f++) {
                    model->w[f] -= model->lr * (2 * model->lambda * model->w[f] - y[i] * X[i][f]);
                }
                model->b -= model->lr * y[i];
            }
        }
    }
}

int predict_svm(SVM *model, double *x) {
    double linear_output = dot_product(x, model->w, model->n_features) - model->b;
    return (linear_output >= 0) ? 1 : -1;
}

void free_svm(SVM *model) {
    free(model->w);
    free(model);
}