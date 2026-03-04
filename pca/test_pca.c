#include "pca.h"

int main() {
    int n_samples = 5;
    int n_features = 2;
    double **X = (double**)malloc(n_samples * sizeof(double*));
    for(int i=0; i<n_samples; i++) X[i] = (double*)malloc(n_features * sizeof(double));

    double raw_data[5][2] = {{1,2}, {2,3}, {3,4}, {4,5}, {5,6}};
    for(int i=0; i<n_samples; i++)
        for(int j=0; j<n_features; j++) X[i][j] = raw_data[i][j];

    PCA *pca = create_pca(1, n_features);
    fit_pca(pca, X, n_samples);

    double **X_proj = (double**)malloc(n_samples * sizeof(double*));
    for(int i=0; i<n_samples; i++) X_proj[i] = (double*)malloc(1 * sizeof(double));

    transform_pca(pca, X, X_proj, n_samples);

    printf("Dati proiettati 1D:\n");
    for(int i=0; i<n_samples; i++) printf("%f\n", X_proj[i][0]);

    return 0;
}