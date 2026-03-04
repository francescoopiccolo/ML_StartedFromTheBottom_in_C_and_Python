#include <stdio.h>
#include <math.h>
#include "logistic.h"

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void fit(double *X, double *y, int n_samples, double *weights, double *bias, double lr, int iterations) {
    for (int iter = 0; iter < iterations; iter++) {
        double dw = 0.0, db = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double z = X[i] * (*weights) + (*bias);
            double y_pred = sigmoid(z);
            double error = y_pred - y[i];
            dw += error * X[i];
            db += error;
        }
        *weights -= lr * (dw / n_samples);
        *bias -= lr * (db / n_samples);
    }
}

int main() {
    double X[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    double y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    double w = 0.0, b = 0.0;

    fit(X, y, 10, &w, &b, 0.1, 1000);

    printf("Modello Allenato!\nPeso (w): %.4f, Bias (b): %.4f\n", w, b);
    
    // Test rapido su un numero (es. 15)
    double test = 15.0;
    printf("Predizione per %.1f: %.4f (Classe: %d)\n", test, sigmoid(test*w + b), (sigmoid(test*w + b) > 0.5));
    
    return 0;
}