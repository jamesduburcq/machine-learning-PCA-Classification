import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time  # <--- Ajouté pour le compteur de temps
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

try:
    from Script01_PreprocessingExploration import compute_hog, my_PCA, TARGET_SIZE , get_resized_db , read_and_crop_db , my_PCA
except ImportError:
    # Safe defaults if utils are missing during initial setup
    TARGET_SIZE = (64, 64)
    def my_PCA(data, n_components=5): pass
    def compute_hog(image, nb_h_cells, nb_w_cells, nb_bins): return np.zeros(nb_h_cells*nb_w_cells*nb_bins)
    def load_dict(f): return {0: 'Chihuahua', 1: 'Pug', 2: 'Malamute', 3: 'Beagle'}

OUTPUT_DIR = os.path.join('..', 'folder_code2')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Define or reload precalculated vector states from Lab 1
if os.path.exists("X_train_standard.npy"):
    X_train = np.load("X_train_standard.npy")
    X_test = np.load("X_test_standard.npy")
    y_train = np.load("y_train_standard.npy")
    y_test = np.load("y_test_standard.npy")
    label_names = load_dict("lbl_names.npy")
    labels = np.load("labels.npy")
else:
    print(" Cache arrays not found. Generating design matrices from raw images...")
    
    bw_imgs, bw_dogs, labels, label_names = read_and_crop_db(color=False)
    
    if bw_imgs is None:
        raise FileNotFoundError("Le dossier 'SmallDB' est introuvable. Vérifiez son chemin d'accès.")
        
    resized_dogs = get_resized_db(bw_dogs, target_size=TARGET_SIZE, pad_type='white')
    
    data_mtx = resized_dogs.reshape(len(resized_dogs), -1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        data_mtx, labels, test_size=0.25, stratify=labels, random_state=42
    )
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    labels = np.array(labels)
    
    print(f" Base de données chargée en direct ! Dimensions de X_train : {X_train.shape}")


# =========================================================================
# TASK 1: QUANTIFYING TRAIN/TEST DISTRIBUTION RESEMBLANCE 
# =========================================================================
def emp_dist(y_train, y_test):
    _ , occ_train = np.unique(y_train,return_counts = True)
    _ , occ_test = np.unique(y_test,return_counts = True)
    
    prop_train = occ_train / sum(occ_train)
    prop_test = occ_test / sum(occ_test)
    return prop_train , prop_test

def cos_sim(prop_train,prop_test):
    return np.dot(prop_train,prop_test) / (np.linalg.norm(prop_train) * np.linalg.norm(prop_test))

prop_train,prop_test = emp_dist(y_train,y_test)
similarity_score = cos_sim(prop_train, prop_test)
print(f"   - Train/Test Resemblance (Cosine Similarity): {similarity_score:.6f}")


# =========================================================================
# TASK 2: CUSTOM PIPELINE TRANSFORMER WRAPPERS 
# =========================================================================
class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, nb_h_cells=4, nb_w_cells=4, nb_bins=8):
        self.nb_h_cells = nb_h_cells
        self.nb_w_cells = nb_w_cells
        self.nb_bins = nb_bins
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return np.array([compute_hog(img.reshape(TARGET_SIZE), self.nb_h_cells, self.nb_w_cells, self.nb_bins) for img in X])


class PCAInfoPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca_per_class = []
        
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Supervised target array 'y' is required to fit class-specific subspaces.")
        
        self.pca_per_class = []
        for race in np.unique(y):
            self.pca_per_class.append(my_PCA(X[y == race], n_components=self.n_components))
        return self
        
    def transform(self, X):
        out_list = []
        for p in self.pca_per_class:
            out_list.append(p.transform(X))
        out = np.hstack(out_list)
        return out


# =========================================================================
# MODIFICATION 2 : GÉNÉRATION ALÉATOIRE CIBLÉE PAR CLASSE
# =========================================================================
def generer_chien_par_classe(X_train, y_train, classe_cible, n_components=15, target_size=(64, 64)):
    """
    Génère un chien synthétique basé uniquement sur l'espace latent d'une classe précise.
    Estimation : 15 composantes suffisent pour capturer l'essentiel des traits d'une seule race.
    """
    X_classe = X_train[y_train == classe_cible]
    
    if len(X_classe) == 0:
        raise ValueError(f"La classe {classe_cible} n'existe pas dans le jeu d'entraînement.")
    
    n_components = min(n_components, len(X_classe) - 1)
    
    pca_classe = PCA(n_components=n_components, random_state=42)
    pca_classe.fit(X_classe)
    
    vecteur_aleatoire = np.random.randn(n_components)
    ecarts_types = np.sqrt(pca_classe.explained_variance_)
    vecteur_latent_calibre = vecteur_aleatoire * ecarts_types
    
    image_plate = pca_classe.inverse_transform(vecteur_latent_calibre.reshape(1, -1))
    image_chien_genere = image_plate.reshape(target_size)
    
    return image_chien_genere


# Construction globale des Features et de la Pipeline
all_features = FeatureUnion([
    ('pca_features', PCAInfoPreprocessing(n_components=5)),
    ('hog_features', EdgeInfoPreprocessing(nb_h_cells=4, nb_w_cells=4, nb_bins=8))
])

pipeline_svc = Pipeline([
    ('minmax_scaler', MinMaxScaler()),
    ('features', all_features),
    ('std_scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf', random_state=42))
])


# =========================================================================
# TASK 4: HYPERPARAMETER OPTIMIZATION
# =========================================================================
print("\n Step 3: Tuning via GridSearchCV...")
if pipeline_svc is not None:
    param_grid = {
        'features__pca_features__n_components': [5, 10],
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': [0.01, 0.1]
    }
    
    grid_search = GridSearchCV(pipeline_svc, param_grid=param_grid, cv=3, n_jobs=-1)
    print("   Training the Grid Search ")
    grid_search.fit(X_train, y_train)
    print(f"    Optimal Parameters Identified: {grid_search.best_params_}")
    print(f"    Generalization Score on Testing Partition: {grid_search.score(X_test, y_test)*100:.2f}%")


# =========================================================================
# TASK 5 & MODIFICATION 1 : COMPARAISON OvO vs. OvR AVEC PARAMÈTRES OPTIMAUX
# =========================================================================
print("\n Step 4: Comparing OvO vs. OvR avec les paramètres optimaux")

best_params = grid_search.best_params_
best_n_components = best_params['features__pca_features__n_components']
best_C = best_params['classifier__C']
best_gamma = best_params['classifier__gamma']

features_optimales = FeatureUnion([
    ('pca_features', PCAInfoPreprocessing(n_components=best_n_components)),
    ('hog_features', EdgeInfoPreprocessing(nb_h_cells=4, nb_w_cells=4, nb_bins=8))
])

svc_optimal = SVC(kernel='rbf', C=best_C, gamma=best_gamma, random_state=42)

pipeline_ovo = Pipeline([
    ('minmax_scaler', MinMaxScaler()),
    ('features', features_optimales),
    ('std_scaler', StandardScaler()),
    ('classifier', OneVsOneClassifier(svc_optimal))
])

pipeline_ovr = Pipeline([
    ('minmax_scaler', MinMaxScaler()),
    ('features', features_optimales),
    ('std_scaler', StandardScaler()),
    ('classifier', OneVsRestClassifier(svc_optimal))
])

start_ovo = time.time()
pipeline_ovo.fit(X_train, y_train)
ovo_score = pipeline_ovo.score(X_test, y_test)
execution_time_ovo = time.time() - start_ovo

start_ovr = time.time()
pipeline_ovr.fit(X_train, y_train)
ovr_score = pipeline_ovr.score(X_test, y_test)
execution_time_ovr = time.time() - start_ovr

# Affichage des résultats
print(f"   One-vs-One (OvO) Strategy Score: {ovo_score*100:.2f}% | Temps: {execution_time_ovo:.4f}s")
print(f"   One-vs-Rest (OvR) Strategy Score: {ovr_score*100:.2f}% | Temps: {execution_time_ovr:.4f}s")

# =========================================================================
# AFFICHAGE DES MATRICES DE CONFUSION (OvO vs. OvR)
# =========================================================================
print("\n Génération des matrices de confusion...")


y_pred_ovo = pipeline_ovo.predict(X_test)
y_pred_ovr = pipeline_ovr.predict(X_test)


classes_ordered = [label_names[i] for i in sorted(label_names.keys())]


fig, ax = plt.subplots(1, 2, figsize=(14, 6))


ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred_ovo, 
    display_labels=classes_ordered,
    cmap=plt.cm.Blues,
    ax=ax[0]
)
ax[0].set_title(f"Matrice de Confusion - OvO\n(Score: {ovo_score*100:.2f}%)")

ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred_ovr, 
    display_labels=classes_ordered,
    cmap=plt.cm.Greens, # Une couleur différente pour bien distinguer
    ax=ax[1]
)
ax[1].set_title(f"Matrice de Confusion - OvR\n(Score: {ovr_score*100:.2f}%)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'comparaison_matrices_confusion.png'))
plt.show()