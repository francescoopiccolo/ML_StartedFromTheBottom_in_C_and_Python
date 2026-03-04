#include "linear_reg.h"

LinearRegression* create_model(int n_features, double lr, int iters, double l1, double l2) {
    LinearRegression *model = (LinearRegression*)malloc(sizeof(LinearRegression));
    model->n_features = n_features;
    model->lr = lr;
    model->iters = iters;
    model->l1_pen = l1;
    model->l2_pen = l2;
    model->w = (double*)calloc(n_features, sizeof(double));
    model->b = 0.0;
    return model;
}

double sign(double x) {
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    return 0.0;
}

void fit(LinearRegression *model, double **X, double *y, int n_samples) {
    int nf = model->n_features;
    
    for (int iter = 0; iter < model->iters; iter++) {
        double *dw = (double*)calloc(nf, sizeof(double));
        double db = 0.0;

        for (int i = 0; i < n_samples; i++) {
            // 1. Predizione: y_hat = Xw + b
            double y_hat = model->b;
            for (int j = 0; j < nf; j++) {
                y_hat += X[i][j] * model->w[j];
            }

            // 2. Errore (y_hat - y)
            double error = y_hat - y[i];

            // 3. Gradiente base MSE
            for (int j = 0; j < nf; j++) {
                dw[j] += (1.0 / n_samples) * error * X[i][j];
            }
            db += (1.0 / n_samples) * error;
        }

        // 4. Aggiunta Penalità Ridge e Lasso
        for (int j = 0; j < nf; j++) {
            // Derivata L2: lambda2 * 2 * w
            dw[j] += (model->l2_pen * 2.0 * model->w[j]) / n_samples;
            
            // Derivata L1: lambda1 * sign(w)
            dw[j] += (model->l1_pen * sign(model->w[j])) / n_samples;

            // 5. Aggiornamento Pesi
            model->w[j] -= model->lr * dw[j];
        }
        model->b -= model->lr * db;

        free(dw);
    }
}

double predict(LinearRegression *model, double *x) {
    double res = model->b;
    for (int i = 0; i < model->n_features; i++) {
        res += x[i] * model->w[i];
    }
    return res;
}

void free_model(LinearRegression *model) {
    free(model->w);
    free(model);
}