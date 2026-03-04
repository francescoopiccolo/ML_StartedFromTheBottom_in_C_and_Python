#include "naive_bayes.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

NaiveBayes* create_nb(int n_classes, int n_features) {
    NaiveBayes *model = (NaiveBayes*)malloc(sizeof(NaiveBayes));
    model->n_classes = n_classes;
    model->n_features = n_features;
    
    model->means = (double**)malloc(n_classes * sizeof(double*));
    model->vars = (double**)malloc(n_classes * sizeof(double*));
    for (int i = 0; i < n_classes; i++) {
        model->means[i] = (double*)calloc(n_features, sizeof(double));
        model->vars[i] = (double*)calloc(n_features, sizeof(double));
    }
    
    model->priors = (double*)calloc(n_classes, sizeof(double));
    model->classes = (int*)malloc(n_classes * sizeof(int));
    return model;
}

double gaussian_pdf(double x, double mean, double var) {
    double eps = 1e-10; // Per stabilità numerica
    double exponent = exp(-(pow(x - mean, 2)) / (2 * var + eps));
    return (1.0 / sqrt(2 * M_PI * var + eps)) * exponent;
}

void fit_nb(NaiveBayes *model, double **X, int *y, int n_samples) {
    // Supponiamo classi 0, 1, 2... per semplicità
    for (int c = 0; c < model->n_classes; c++) {
        model->classes[c] = c;
        int count = 0;
        
        // 1. Calcolo Media
        for (int i = 0; i < n_samples; i++) {
            if (y[i] == c) {
                for (int f = 0; f < model->n_features; f++) {
                    model->means[c][f] += X[i][f];
                }
                count++;
            }
        }
        
        for (int f = 0; f < model->n_features; f++) {
            model->means[c][f] /= count;
        }
        model->priors[c] = (double)count / n_samples;

        // 2. Calcolo Varianza
        for (int i = 0; i < n_samples; i++) {
            if (y[i] == c) {
                for (int f = 0; f < model->n_features; f++) {
                    model->vars[c][f] += pow(X[i][f] - model->means[c][f], 2);
                }
            }
        }
        for (int f = 0; f < model->n_features; f++) {
            model->vars[c][f] /= count;
        }
    }
}

int predict_nb(NaiveBayes *model, double *x) {
    double best_posterior = -INFINITY;
    int best_class = -1;

    for (int c = 0; c < model->n_classes; c++) {
        double posterior = log(model->priors[c]);
        for (int f = 0; f < model->n_features; f++) {
            double pdf_val = gaussian_pdf(x[f], model->means[c][f], model->vars[c][f]);
            posterior += log(pdf_val + 1e-10); // Log-trick
        }

        if (posterior > best_posterior) {
            best_posterior = posterior;
            best_class = model->classes[c];
        }
    }
    return best_class;
}

void free_nb(NaiveBayes *model) {
    for (int i = 0; i < model->n_classes; i++) {
        free(model->means[i]);
        free(model->vars[i]);
    }
    free(model->means);
    free(model->vars);
    free(model->priors);
    free(model->classes);
    free(model);
}