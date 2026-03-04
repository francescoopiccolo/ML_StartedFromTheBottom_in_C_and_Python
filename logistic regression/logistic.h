#ifndef LOGISTIC_H
#define LOGISTIC_H

// Prototipi delle funzioni
double sigmoid(double z);
void fit(double *X, double *y, int n_samples, double *weights, double *bias, double lr, int iterations);

#endif