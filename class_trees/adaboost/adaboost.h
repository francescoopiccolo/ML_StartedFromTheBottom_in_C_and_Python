#ifndef ADABOOST_H
#define ADABOOST_H

typedef struct {
    int feature_idx;
    double threshold;
    int polarity;
    double alpha;
} DecisionStump;

typedef struct {
    int n_stumps;
    DecisionStump *stumps;
} AdaBoost;

// Funzioni principali
AdaBoost* create_adaboost(int n_stumps);
void train_adaboost(AdaBoost *model, double **X, int *y, int n_samples, int n_features);
int predict_adaboost(AdaBoost *model, double *x);
void free_adaboost(AdaBoost *model);

#endif