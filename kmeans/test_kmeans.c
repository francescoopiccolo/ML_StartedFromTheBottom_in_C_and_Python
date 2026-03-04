#include "kmeans.h"

int main() {
    int n_samples = 9;
    int n_features = 2;
    int k = 3;

    double **X = (double**)malloc(n_samples * sizeof(double*));
    for (int i = 0; i < n_samples; i++) X[i] = (double*)malloc(n_features * sizeof(double));

    // Tre gruppi chiari: (1,1), (5,5), (1,5)
    double raw_data[9][2] = {
        {1.1, 1.2}, {0.9, 0.8}, {1.0, 1.0}, // Cluster 1
        {4.9, 5.1}, {5.0, 5.0}, {5.2, 4.8}, // Cluster 2
        {0.8, 5.2}, {1.2, 4.9}, {1.0, 5.0}  // Cluster 3
    };

    for(int i=0; i<n_samples; i++)
        for(int j=0; j<n_features; j++) X[i][j] = raw_data[i][j];

    KMeans *km = create_kmeans(k, n_features, 100);
    fit_kmeans(km, X, n_samples);

    printf("Centroidi finali:\n");
    for(int i=0; i<k; i++) {
        printf("C%d: [%.2f, %.2f]\n", i, km->centroids[i][0], km->centroids[i][1]);
    }

    double test_point[] = {1.1, 0.9};
    printf("Il punto [1.1, 0.9] appartiene al cluster: %d\n", predict_kmeans(km, test_point));

    // Pulizia memoria
    for(int i=0; i<n_samples; i++) free(X[i]);
    free(X);
    free_kmeans(km);

    return 0;
}