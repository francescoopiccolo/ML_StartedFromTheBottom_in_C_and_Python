#ifndef GRADIENT_BOOSTING_H
#define GRADIENT_BOOSTING_H

typedef struct {
    int feature_idx;
    double threshold;
    double left_val;
    double right_val;
} RegressionStump;

typedef struct {
    int n_estimators;
    double lr;
    double init_pred;
    RegressionStump *trees;
} GradientBoosting;

GradientBoosting* create_gb(int n_estimators, double lr);
void train_gb(GradientBoosting *model, double **X, int *y, int n_samples, int n_features);
int predict_gb(GradientBoosting *model, double *x);
void free_gb(GradientBoosting *model);

#endif