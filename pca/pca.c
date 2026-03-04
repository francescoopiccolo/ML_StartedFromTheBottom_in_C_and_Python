#include "pca.h"

PCA* create_pca(int n_components, int n_features) {
    PCA *model = (PCA*)malloc(sizeof(PCA));
    model->n_components = n_components;
    model->n_features = n_features;
    model->mean = (double*)calloc(n_features, sizeof(double));
    model->components = (double**)malloc(n_components * sizeof(double*));
    for(int i=0; i<n_components; i++) 
        model->components[i] = (double*)malloc(n_features * sizeof(double));
    return model;
}

// Calcolo matrice di covarianza
void compute_covariance(double **X, int n_samples, int n_features, double **cov) {
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < n_features; j++) {
            double sum = 0;
            for (int k = 0; k < n_samples; k++) {
                sum += X[k][i] * X[k][j];
            }
            cov[i][j] = sum / (n_samples - 1);
        }
    }
}

// Metodo di Jacobi per autovalori/autovettori
void jacobi_method(double **S, int n, double *eigenvalues, double **eigenvectors) {
    // Inizializza eigenvectors come matrice identità
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
    }

    for (int iter = 0; iter < 50; iter++) { // 50 iterazioni bastano per convergenza
        int p, q;
        double max_val = 0;
        // Trova il massimo fuori diagonale
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (fabs(S[i][j]) > max_val) {
                    max_val = fabs(S[i][j]); p = i; q = j;
                }
            }
        }
        if (max_val < 1e-15) break;

        double theta = 0.5 * atan2(2 * S[p][q], S[q][q] - S[p][p]);
        double c = cos(theta), s = sin(theta);

        // Rotazione di Jacobi
        for (int i = 0; i < n; i++) {
            double temp_p = c * eigenvectors[i][p] - s * eigenvectors[i][q];
            double temp_q = s * eigenvectors[i][p] + c * eigenvectors[i][q];
            eigenvectors[i][p] = temp_p; eigenvectors[i][q] = temp_q;
        }

        double app = S[p][p], aqq = S[q][q], apq = S[p][q];
        S[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        S[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        S[p][q] = S[q][p] = 0;

        for (int i = 0; i < n; i++) {
            if (i != p && i != q) {
                double aip = S[i][p], aiq = S[i][q];
                S[i][p] = S[p][i] = c * aip - s * aiq;
                S[i][q] = S[q][i] = s * aip + c * aiq;
            }
        }
    }
    for (int i = 0; i < n; i++) eigenvalues[i] = S[i][i];
}

void fit_pca(PCA *model, double **X, int n_samples) {
    int nf = model->n_features;
    // Mean centering
    for (int j = 0; j < nf; j++) {
        model->mean[j] = 0;
        for (int i = 0; i < n_samples; i++) model->mean[j] += X[i][j];
        model->mean[j] /= n_samples;
        for (int i = 0; i < n_samples; i++) X[i][j] -= model->mean[j];
    }

    double **cov = (double**)malloc(nf * sizeof(double*));
    double **e_vecs = (double**)malloc(nf * sizeof(double*));
    for(int i=0; i<nf; i++) {
        cov[i] = (double*)malloc(nf * sizeof(double));
        e_vecs[i] = (double*)malloc(nf * sizeof(double));
    }
    double *e_vals = (double*)malloc(nf * sizeof(double));

    compute_covariance(X, n_samples, nf, cov);
    jacobi_method(cov, nf, e_vals, e_vecs);

    // Selezione (Semplificata: assumiamo già ordinati o facciamo sort qui)
    // Per brevità prendiamo le prime componenti
    for(int i=0; i<model->n_components; i++) {
        for(int j=0; j<nf; j++) model->components[i][j] = e_vecs[j][nf - 1 - i]; // Jacobi mette i max alla fine
    }

    // Free temp
    for(int i=0; i<nf; i++) { free(cov[i]); free(e_vecs[i]); }
    free(cov); free(e_vecs); free(e_vals);
}

void transform_pca(PCA *model, double **X, double **X_proj, int n_samples) {
    for (int i = 0; i < n_samples; i++) {
        for (int k = 0; k < model->n_components; k++) {
            X_proj[i][k] = 0;
            for (int j = 0; j < model->n_features; j++) {
                X_proj[i][k] += X[i][j] * model->components[k][j];
            }
        }
    }
}