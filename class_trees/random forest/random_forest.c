#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "random_forest.h"

// Funzione statica (privata) per l'entropia
static double calculate_entropy(int *y, int n) {
    if (n == 0) return 0;
    int counts[2] = {0, 0};
    for (int i = 0; i < n; i++) counts[y[i]]++;
    double entropy = 0;
    for (int i = 0; i < 2; i++) {
        double p = (double)counts[i] / n;
        if (p > 0) entropy -= p * log2(p);
    }
    return entropy;
}

static Node* create_node(int val) {
    Node *node = (Node*)malloc(sizeof(Node));
    node->feature_idx = -1;
    node->threshold = 0;
    node->value = val;
    node->left = node->right = NULL;
    return node;
}

// Funzione ricorsiva per far crescere l'albero
static Node* grow_tree(double **X, int *y, int n_samples, int n_features, int n_f_split, int depth, int max_depth) {
    int sum_y = 0;
    for(int i=0; i<n_samples; i++) sum_y += y[i];
    
    if (depth >= max_depth || sum_y == 0 || sum_y == n_samples || n_samples < 2) {
        return create_node(sum_y > n_samples / 2 ? 1 : 0);
    }

    double best_ig = -1;
    int best_f = -1;
    double best_t = 0;

    // Feature Selection: scegliamo n_f_split indici a caso
    for (int k = 0; k < n_f_split; k++) {
        int f = rand() % n_features; 
        for (int i = 0; i < n_samples; i++) {
            double threshold = X[i][f];
            int n_l = 0, n_r = 0;
            for(int j=0; j<n_samples; j++) {
                if (X[j][f] <= threshold) n_l++; else n_r++;
            }
            if (n_l == 0 || n_r == 0) continue;

            int *y_l = malloc(n_l * sizeof(int));
            int *y_r = malloc(n_r * sizeof(int));
            int il = 0, ir = 0;
            for(int j=0; j<n_samples; j++) {
                if (X[j][f] <= threshold) y_l[il++] = y[j];
                else y_r[ir++] = y[j];
            }

            double ig = calculate_entropy(y, n_samples) - 
                        (((double)n_l/n_samples)*calculate_entropy(y_l, n_l) + 
                         ((double)n_r/n_samples)*calculate_entropy(y_r, n_r));

            if (ig > best_ig) {
                best_ig = ig; best_f = f; best_t = threshold;
            }
            free(y_l); free(y_r);
        }
    }

    if (best_f == -1) return create_node(sum_y > n_samples / 2 ? 1 : 0);

    Node *node = create_node(-1);
    node->feature_idx = best_f;
    node->threshold = best_t;

    int n_l = 0, n_r = 0;
    for(int i=0; i<n_samples; i++) if (X[i][best_f] <= best_t) n_l++; else n_r++;
    
    double **X_l = malloc(n_l * sizeof(double*));
    int *y_l = malloc(n_l * sizeof(int));
    double **X_r = malloc(n_r * sizeof(double*));
    int *y_r = malloc(n_r * sizeof(int));

    int il = 0, ir = 0;
    for(int i=0; i<n_samples; i++) {
        if (X[i][best_f] <= best_t) { X_l[il] = X[i]; y_l[il++] = y[i]; }
        else { X_r[ir] = X[i]; y_r[ir++] = y[i]; }
    }

    node->left = grow_tree(X_l, y_l, n_l, n_features, n_f_split, depth + 1, max_depth);
    node->right = grow_tree(X_r, y_r, n_r, n_features, n_f_split, depth + 1, max_depth);

    free(X_l); free(y_l); free(X_r); free(y_r);
    return node;
}

RandomForest* create_rf(int n_trees, int max_depth, int n_features_split) {
    RandomForest *rf = malloc(sizeof(RandomForest));
    rf->n_trees = n_trees;
    rf->max_depth = max_depth;
    rf->n_features_split = n_features_split;
    rf->roots = malloc(n_trees * sizeof(Node*));
    return rf;
}

void train_rf(RandomForest *rf, double **X, int *y, int n_samples, int n_features) {
    for (int i = 0; i < rf->n_trees; i++) {
        double **X_boot = malloc(n_samples * sizeof(double*));
        int *y_boot = malloc(n_samples * sizeof(int));
        for (int j = 0; j < n_samples; j++) {
            int idx = rand() % n_samples;
            X_boot[j] = X[idx];
            y_boot[j] = y[idx];
        }
        rf->roots[i] = grow_tree(X_boot, y_boot, n_samples, n_features, rf->n_features_split, 0, rf->max_depth);
        free(X_boot); free(y_boot);
    }
}

static int predict_tree(Node *node, double *x) {
    if (node->value != -1) return node->value;
    if (x[node->feature_idx] <= node->threshold) return predict_tree(node->left, x);
    return predict_tree(node->right, x);
}

int predict_rf(RandomForest *rf, double *x) {
    int v0 = 0, v1 = 0;
    for (int i = 0; i < rf->n_trees; i++) {
        if (predict_tree(rf->roots[i], x) == 0) v0++; else v1++;
    }
    return (v1 > v0) ? 1 : 0;
}