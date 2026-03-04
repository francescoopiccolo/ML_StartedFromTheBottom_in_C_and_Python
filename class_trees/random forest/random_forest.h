#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

// Struttura del Nodo dell'albero
typedef struct Node {
    int feature_idx;
    double threshold;
    int value;           // -1 se nodo interno, 0 o 1 se foglia
    struct Node *left;
    struct Node *right;
} Node;

// Struttura della Random Forest
typedef struct {
    int n_trees;
    int max_depth;
    int n_features_split; // Numero di feature da considerare a ogni split
    Node **roots;
} RandomForest;

// Prototipi delle funzioni
RandomForest* create_rf(int n_trees, int max_depth, int n_features_split);
void train_rf(RandomForest *rf, double **X, int *y, int n_samples, int n_features);
int predict_rf(RandomForest *rf, double *x);
void free_rf(RandomForest *rf); // Per non lasciare memory leak

#endif