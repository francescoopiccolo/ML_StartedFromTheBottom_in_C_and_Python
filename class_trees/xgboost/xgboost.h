#ifndef XGBOOST_H
#define XGBOOST_H

typedef struct {
    int feature_idx;
    double threshold;
    double left_val;
    double right_val;
    double lambd;
} XGBoostTree;

typedef struct {
    int n_estimators;
    double lr;
    double lambd;
    double init_log_odds;
    XGBoostTree *trees;
} XGBoost;

XGBoost* create_xgb(int n_estimators, double lr, double lambd);
void train_xgb(XGBoost *model, double **X, int *y, int n_samples, int n_features);
double predict_proba_xgb(XGBoost *model, double *x);
int predict_xgb(XGBoost *model, double *x);
void free_xgb(XGBoost *model);

#endif