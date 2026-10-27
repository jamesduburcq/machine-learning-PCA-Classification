import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import time

# Ensure you have your utils.py file or these functions available in your environment
try:
    from Script01_PreprocessingExploration_Orion import compute_hog_hsv, compute_hog, my_PCA, TARGET_SIZE , get_resized_db , read_and_crop_db , my_PCA
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
   
    # 1. Charger et détourer (crop) les images depuis le dossier SmallDB
    bw_imgs, bw_dogs, labels, label_names = read_and_crop_db(color=True)
   
    if bw_imgs is None:
        raise FileNotFoundError("Le dossier 'SmallDB' est introuvable. Vérifiez son chemin d'accès.")
       
    # 2. Redimensionner les visages de chiens avec le format cible (TARGET_SIZE)
    resized_dogs = get_resized_db(bw_dogs, target_size=TARGET_SIZE, pad_type='continuous')
   
    # 3. Aplatir les images 2D (N, H, W) en une matrice de caractéristiques 2D (N, H*W)
    data_mtx = resized_dogs.reshape(len(resized_dogs), -1)
   
    # 4. Partitionner les données en jeu d'entraînement et de test (Stratifié)
    X_train, X_test, y_train, y_test = train_test_split(
        data_mtx, labels, test_size=0.25, stratify=labels, random_state=42
    )
   
    # Convertir les listes de labels en tableaux NumPy pour la suite des tâches
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    labels = np.array(labels)
   
    print(f" Base de données chargée en direct ! Dimensions de X_train : {X_train.shape}")
   
    print("calcul des proportions par ensembles et traçage du graph")
    classes_global, counts_global = np.unique(labels, return_counts=True)
    prop_global = counts_global / len(labels)

    classes_train, counts_train = np.unique(y_train, return_counts=True)
    prop_train = counts_train / len(y_train)

    classes_test, counts_test = np.unique(y_test, return_counts=True)
    prop_test = counts_test / len(y_test)
   
    races = ['Chihuahua', 'basset', 'Kerry_blue_terrier', 'groenendael', 'malinois', 'chow']

    # Les proportions que tu as calculées (ici des valeurs d'exemple)
   
    # 2. Paramétrage de l'axe des X
    x = np.arange(len(races))  # Crée les positions de base : [0, 1, 2, 3, 4, 5]
    largeur = 0.2              # La largeur de chaque bâton
   
    # 3. Création de la figure
    fig, ax = plt.subplots(figsize=(8, 5))
   
    # 4. Tracé des trois séries de barres avec les couleurs correspondantes
    # On décale le train vers la gauche (-largeur), le test au centre, l'original vers la droite (+largeur)
    ax.bar(x - largeur, prop_train, largeur, label='Training Split', color='#538bc2')
    ax.bar(x, prop_test, largeur, label='Testing Split', color='#cb3335')
    ax.bar(x + largeur, prop_global, largeur, label='Original Population', color='#bfbfbf')
   
    # 5. Habillage du graphique (titres, labels)
    ax.set_ylabel('Empirical Class Probabilities')
    ax.set_title('Dataset Distribution Mapping Analysis')
   
    # On place les noms des races exactement sur les positions "x" (au centre du groupe de 3 barres)
    ax.set_xticks(x)
    ax.set_xticklabels(races, rotation=15, ha='right', fontsize=11)
   
    # Ajout de la légende
    ax.legend()
   
    # Ajuste les marges pour que les labels rotatifs ne soient pas coupés
    plt.tight_layout()
   
    # Affichage ou sauvegarde
    # plt.savefig('figures/distribution_mapping.png', dpi=150)
    plt.show()
#%%


# =========================================================================
# TASK 1: QUANTIFYING TRAIN/TEST DISTRIBUTION RESEMBLANCE
# =========================================================================
# Prevent the "Time Machine Effect". We must ensure our random splits
# have similar class distributions before applying any transformations.

### STUDENT IMPLEMENTATION START ###
# TODO: 1. Calculate the empirical distribution arrays (class probabilities) for y_train and y_test.
#          Hint: Use np.unique(..., return_counts=True) and divide by the total length.
# TODO: 2. Implement the Cosine Similarity metric to quantify the resemblance:
#          Formula: (A dot B) / (norm(A) * norm(B))
# What other similarity metrics could you have used here? (e.g., KL Divergence, Wasserstein Distance, etc.) How would you implement them mathematically?

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
   
   
   

### STUDENT IMPLEMENTATION END ###
print(f"   - Train/Test Resemblance (Cosine Similarity): {similarity_score:.6f}")


# =========================================================================
# TASK 2: CUSTOM PIPELINE TRANSFORMER WRAPPERS
# =========================================================================
# The "Lego Block" Philosophy: We wrap our custom functions into classes
# that inherit from BaseEstimator and TransformerMixin.

class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, nb_h_cells=4, nb_w_cells=4, nb_bins=8):
        self.nb_h_cells = nb_h_cells
        self.nb_w_cells = nb_w_cells
        self.nb_bins = nb_bins
       
    def fit(self, X, y=None):
        return self # HOG requires no training, so fit does nothing
       
    def transform(self, X):
        return np.array([compute_hog_hsv(img.reshape(TARGET_SIZE[0],TARGET_SIZE[1],3), self.nb_h_cells, self.nb_w_cells, self.nb_bins) for img in X])


class PCAInfoPreprocessing(BaseEstimator, TransformerMixin):
    """
    In Module 1, class-specific PCA was computed manually via loops.
    To prevent Data Leakage, we automate this inside a scikit-learn transformer.
    """
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca_per_class = []
       
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Supervised target array 'y' is required to fit class-specific subspaces.")
       
        ### STUDENT IMPLEMENTATION START ###
        # TODO: 1. Clear historical instances tracked in self.pca_per_class.
        # TODO: 2. Iterate through unique class identifiers present within vector 'y'.
        # TODO: 3. Isolate matching instance data slices from 'X', train separate PCA models,
        #          and append each fitted model to self.pca_per_class.
       
        self.pca_per_class = []
        for race in np.unique(y):
            self.pca_per_class.append(my_PCA(  X[y == race]  ,  n_components=self.n_components  ))
       
       
        ### STUDENT IMPLEMENTATION END ###
        return self
       
    def transform(self, X):
       
        out = np.zeros((len(X), 0)) # Optionnel : conservé du template initial
       
        ### STUDENT IMPLEMENTATION START ###
        out_list = []
       
       
        # TODO: 1. Iterate through the fitted PCA models saved in self.pca_per_class.
        # TODO: 2. Transform the input matrix 'X' using each PCA model.
        # TODO: 3. Horizontally stack the outputs together using np.hstack.
       
        for p in self.pca_per_class:
            out_list.append(p.transform(X))
           
        out = np.hstack(out_list)
        ### STUDENT IMPLEMENTATION END ###
        return out



### STUDENT IMPLEMENTATION START ###
# TODO: 1. Construct a 'FeatureUnion' merging 'PCAInfoPreprocessing' and 'EdgeInfoPreprocessing'.
#          Name it 'all_features'.
all_features = FeatureUnion([
    ('pca_features', PCAInfoPreprocessing(n_components=5)),
    ('hog_features', EdgeInfoPreprocessing(nb_h_cells=4, nb_w_cells=4, nb_bins=8))
])

# TODO: 2. Build the end-to-end processing pipeline using scikit-learn's 'Pipeline' class.
#          The sequence MUST be: MinMaxScaler -> FeatureUnion -> StandardScaler -> SVC(kernel='linear').
#          Name it 'pipeline_svc'.
pipeline_svc = Pipeline([
    ('minmax_scaler', MinMaxScaler()),
    ('features', all_features),
    ('std_scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf'))
])
### STUDENT IMPLEMENTATION END ###



# =========================================================================
# TASK 4: HYPERPARAMETER OPTIMIZATION SELECTION SPACE (See Slide 6)
# =========================================================================
print("\n Step 3: Tuning via GridSearchCV...")

if pipeline_svc is not None:
    # Switch to RBF kernel for complex boundary mapping
    pipeline_svc.set_params(classifier__kernel='rbf')
   
    ### STUDENT IMPLEMENTATION START ###
    # TODO: Set up a dictionary search parameter grid matching your pipeline's exact component names.
    # Grid Requirements:
    # - PCA components: [5, 10]
    # - SVC Cost C: [0.1, 1, 10]
    # - SVC Gamma: [0.01, 0.1]
   
    param_grid = {'features__pca_features__n_components' : [15],
                  'classifier__C' : [100],
                  'classifier__gamma' : ['scale']
                  }
   
    ### STUDENT IMPLEMENTATION END ###
   
    # We use 3-Fold Cross Validation to test the parameters safely
    grid_search = GridSearchCV(pipeline_svc, param_grid=param_grid, cv=3, n_jobs=-1)
   
    print("   Training the Grid Search ")
   
    grid_search.fit(X_train, y_train)
    print(f"    Optimal Parameters Identified: {grid_search.best_params_}")
    print(f"    Generalization Score on Testing Partition: {grid_search.score(X_test, y_test)*100:.2f}%")


# =========================================================================
# TASK 5: MULTICLASS STRATEGY COMPARISON (See Slides 7 & 8)
# =========================================================================
# Support Vector Machines are binary. Let's compare the "Round Robin" (OvO)
# strategy against the "Me Against the World" (OvR) strategy.
print("\n Step 4: Comparing  OvO vs. OvR ")

# TODO: 1. Build a new pipeline named 'pipeline_ovo'. Use the same scaling and feature
#          steps as before, but wrap the final SVC inside a OneVsOneClassifier().
           
# TODO: 2. Fit 'pipeline_ovo' on the training data and score it on the test data.
# TODO: 3. Repeat the exact same process for a new pipeline named 'pipeline_ovr',
#          but use a OneVsRestClassifier() instead.

# --- OvO Implementation ---
pipeline_ovo = Pipeline([
    ('minmax_scaler', MinMaxScaler()),
    ('features', all_features),
    ('std_scaler', StandardScaler()),
    ('classifier', OneVsOneClassifier(SVC(kernel='rbf',random_state=42)))
])
print("\n Step 3: Tuning via Ovo..")
       
t01=time.perf_counter()
pipeline_ovo.fit(X_train, y_train)
t02=time.perf_counter()

t_OvO=t02-t01

print(f"la methode OvO a mis {t_OvO} secondes")
ovo_score = pipeline_ovo.score(X_test, y_test)

print("\n Step 7: Generating Confusion Matrix")
 # 1. Prédire les classes sur le jeu de test
 # (Pense à remplacer X_test_scaled par X_test_combined si tu désactives le scaler pour la Task 4)
y_pred = pipeline_ovo.predict(X_test)

 # 2. Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

 # 3. Récupérer le nom des races de chien dans l'ordre des labels
target_names = [label_names[i] for i in sorted(label_names.keys())]

 # 4. Affichage et sauvegarde de la matrice
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

 # On plot avec un dégradé de bleu et une rotation des labels pour la lisibilité
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=15)
ax.set_title("Matrice de Confusion par Race de Chien OvO", fontweight='bold', pad=15)

 # Sauvegarde du fichier
plt.show()

# --- OvR Implementation ---
pipeline_ovr = Pipeline([
    ('minmax_scaler', MinMaxScaler()),
    ('features', all_features),
    ('std_scaler', StandardScaler()),
    ('classifier', OneVsRestClassifier((SVC(kernel='rbf', random_state=42))))
])
print("\n Step 3: Tuning via OvR..")

        ### STUDENT IMPLEMENTATION END ###
t1=time.perf_counter()
pipeline_ovr.fit(X_train, y_train)
t2=time.perf_counter()

t_ovr=t2-t1
print(f"la methode OvR a mis {t_ovr} secondes")

ovr_score = pipeline_ovr.score(X_test,y_test)

print("\n Step 7: Generating Confusion Matrix")
 # 1. Prédire les classes sur le jeu de test
 # (Pense à remplacer X_test_scaled par X_test_combined si tu désactives le scaler pour la Task 4)
y_pred = pipeline_ovr.predict(X_test)

 # 2. Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

 # 3. Récupérer le nom des races de chien dans l'ordre des labels
target_names = [label_names[i] for i in sorted(label_names.keys())]

 # 4. Affichage et sauvegarde de la matrice
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

 # On plot avec un dégradé de bleu et une rotation des labels pour la lisibilité
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=15)
ax.set_title("Matrice de Confusion par Race de Chien OvR", fontweight='bold', pad=15)

 # Sauvegarde du fichier
plt.show()

print(f"   One-vs-One (OvO) Strategy Score: {ovo_score*100:.2f}%")
print(f"   One-vs-Rest (OvR) Strategy Score: {ovr_score*100:.2f}%")


# =========================================================================
# TASK 6: PADDING STRATEGY COMPARISON — OvO vs OvR
# =========================================================================
# We evaluate how the choice of padding type ('white', 'black', 'continuous')
# impacts classification accuracy for both multiclass strategies.

print("\n Step 5: Comparing Padding Strategies (white / black / continuous)...")

PAD_TYPES = ['white', 'black', 'continuous']
scores_ovo_pad = []
scores_ovr_pad = []

# Load raw crops once (they are already available from the earlier block)
# If running from cache, re-load raw images to allow repadding
if os.path.exists("X_train_standard.npy"):
    print("   [WARNING] Cache mode active: raw images not available for re-padding.")
    print("   Skipping padding comparison (requires raw image loading path).")
else:
    # bw_dogs and labels are already in memory from the main loading block above
    for pad in PAD_TYPES:
        print(f"   Training with pad_type='{pad}'...")

        # 1. Resize with the current padding strategy
        resized_pad = get_resized_db(bw_dogs, target_size=TARGET_SIZE, pad_type=pad)
        data_pad = resized_pad.reshape(len(resized_pad), -1)

        # 2. Stratified train/test split (same seed for fair comparison)
        X_tr, X_te, y_tr, y_te = train_test_split(
            data_pad, labels, test_size=0.25, stratify=labels, random_state=42
        )
        y_tr = np.array(y_tr)
        y_te = np.array(y_te)

        # 3. Build fresh pipelines (avoid state leakage between iterations)
        pipe_ovo_pad = Pipeline([
            ('minmax_scaler', MinMaxScaler()),
            ('features', FeatureUnion([
                ('pca_features', PCAInfoPreprocessing(n_components=5)),
                ('hog_features', EdgeInfoPreprocessing(nb_h_cells=4, nb_w_cells=4, nb_bins=8))
            ])),
            ('std_scaler', StandardScaler()),
            ('classifier', OneVsOneClassifier(SVC(kernel='rbf', random_state=42)))
        ])

        pipe_ovr_pad = Pipeline([
            ('minmax_scaler', MinMaxScaler()),
            ('features', FeatureUnion([
                ('pca_features', PCAInfoPreprocessing(n_components=5)),
                ('hog_features', EdgeInfoPreprocessing(nb_h_cells=4, nb_w_cells=4, nb_bins=8))
            ])),
            ('std_scaler', StandardScaler()),
            ('classifier', OneVsRestClassifier(SVC(kernel='rbf', random_state=42)))
        ])

        # 4. Fit and score
        pipe_ovo_pad.fit(X_tr, y_tr)
        pipe_ovr_pad.fit(X_tr, y_tr)

        s_ovo = pipe_ovo_pad.score(X_te, y_te)
        s_ovr = pipe_ovr_pad.score(X_te, y_te)

        scores_ovo_pad.append(s_ovo * 100)
        scores_ovr_pad.append(s_ovr * 100)

        print(f"     pad='{pad}' -> OvO: {s_ovo*100:.2f}%  |  OvR: {s_ovr*100:.2f}%")

    # -----------------------------------------------------------------------
    # Histogram: Score vs Padding Type for OvO and OvR
    # -----------------------------------------------------------------------
    x = np.arange(len(PAD_TYPES))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_ovo = ax.bar(x - bar_width / 2, scores_ovo_pad, bar_width,
                      label='OvO (One-vs-One)', color='#2e75b6', edgecolor='black')
    bars_ovr = ax.bar(x + bar_width / 2, scores_ovr_pad, bar_width,
                      label='OvR (One-vs-Rest)', color='#c00000', edgecolor='black')

    # Annotate each bar with its value
    for bar in bars_ovo:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#2e75b6')

    for bar in bars_ovr:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#c00000')

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PAD_TYPES], fontsize=12)
    ax.set_xlabel('Padding Strategy', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Impact of Padding Strategy on Classification Accuracy\n(OvO vs OvR)', fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    pad_fig_path = os.path.join(OUTPUT_DIR, 'padding_comparison.png')
    plt.savefig(pad_fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n   Padding comparison figure saved to '{pad_fig_path}'")