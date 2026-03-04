#include "linear_reg.h"

int main() {
    int n_samples = 5;
    int n_features = 1;

    double **X = (double**)malloc(n_samples * sizeof(double*));
    for (int i = 0; i < n_samples; i++) X[i] = (double*)malloc(n_features * sizeof(double));
    
    double x_vals[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {2.0, 4.0, 5.0, 4.0, 5.0};

    for (int i = 0; i < n_samples; i++) X[i][0] = x_vals[i];

    // Test Ridge
    LinearRegression *ridge = create_model(n_features, 0.01, 1000, 0, 10.0);
    fit(ridge, X, y, n_samples);
    printf("Peso Ridge: %f\n", ridge->w[0]);

    // Test Lasso
    LinearRegression *lasso = create_model(n_features, 0.01, 1000, 10.0, 0);
    fit(lasso, X, y, n_samples);
    printf("Peso Lasso: %f\n", lasso->w[0]);

    // Pulizia
    free_model(ridge);
    free_model(lasso);
    for (int i = 0; i < n_samples; i++) free(X[i]);
    free(X);

    return 0;
}