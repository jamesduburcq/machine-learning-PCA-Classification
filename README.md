# 🐶👤 Classification d'Images & Reconnaissance Faciale par PCA, HOG et SVM

Ce dépôt contient l'implémentation complète d'un projet de Machine Learning et de Vision par Ordinateur réalisé en trinôme (**James, Orion et Raphaël**). 

L'objectif du projet est d'explorer l'extraction de caractéristiques (réduction de dimension par **Analyse en Composantes Principales - PCA** et descripteurs de gradients **HOG**) couplée à des algorithmes de classification supervisée (**KNN** et **SVM multiclasses OvO / OvR**) sur deux problématiques :
1. **La classification de races de chiens** (à partir de la base *Stanford Dogs / SmallDB*) en niveaux de gris et en couleur (espace HSV).
2. **La reconnaissance faciale et la génération de visages synthétiques** sur notre propre base de données de visages.

📄 **Le rapport complet détaillant la méthodologie mathématique et l'analyse des résultats est disponible à la racine du dépôt.**

---

## 📁 Architecture du Projet

Pour garantir le bon fonctionnement des chemins relatifs configurés dans les scripts, le projet s'articule autour de l'arborescence suivante :

```text
📦 Projet-ML-PCA-Classification/
 ┣ 📂 SmallDB/                                     # Dataset des races de chiens
 ┃ ┣ 📂 Images/                                    # Images brutes classées par dossier de race
 ┃ ┗ 📂 Annotation/                                # Annotations XML (Bounding Boxes Pascal VOC)
 ┣ 📂 src/                                         # Scripts Python principaux
 ┃ ┣ 📂 Datasets/
 ┃ ┃ ┗ 📂 att_faces/                               # Photos des visages (dossiers James, Orion, Raphael)
 ┃ ┣ 📂 figures/                                   # Dossier de sortie des visualisations
 ┃ ┣ 📜 Script01_PreprocessingExploration.py       # Partie 1A : Prétraitement, PCA, HOG & KNN (N&B)
 ┃ ┣ 📜 script_SVC.py                              # Partie 1B : Pipeline SVM, GridSearch & OvO vs OvR (N&B)
 ┃ ┣ 📜 Script01_PreprocessingExploration_Orion.py # Partie 2A : Approche Couleur (RGB/HSV), PCA 3D & KNN
 ┃ ┣ 📜 script_2_Orion.py                          # Partie 2B : Pipeline SVM HSV & Étude de l'impact du Padding
 ┃ ┗ 📜 nos_visages.py                             # Partie 3 : Reconnaissance faciale & Génération de visages
 ┣ 📂 folder_code2/                                # Dossier de sortie des comparaisons SVM & Padding
 ┣ 📜 Rapport_Projet.pdf                           # Rapport complet du projet
 ┗ 📜 README.md                                    # Documentation du dépôt
```

---

## ⚙️ Prérequis et Installation

Le projet nécessite **Python 3.8+** ainsi que les bibliothèques scientifiques standards. Vous pouvez les installer via `pip` :

```bash
pip install numpy matplotlib scikit-learn scikit-image seaborn
```

---

## 🚀 Description et Utilisation des Scripts

Tous les scripts doivent être exécutés depuis le dossier `src/` (afin que le chemin relatif `../SmallDB` pointe correctement vers la base de données des chiens et `./Datasets/att_faces` vers celle des visages) :

```bash
cd src
```

### 1. Classification de races de chiens — Approche Niveaux de Gris (Noir & Blanc)

#### 🔹 `Script01_PreprocessingExploration.py`
Ce premier script met en place toute la chaîne de prétraitement et d'analyse exploratoire sur des images redimensionnées en `128x128` en niveaux de gris :
* **Analyse exploratoire :** Calcul de l'entropie de Shannon pour évaluer l'équilibre des classes et affichage de la distribution.
* **Localisation et Prétraitement :** Lecture des fichiers XML d'annotations pour détourer chaque chien (*bounding box*) et redimensionnement conservant le ratio d'aspect (`resize_and_pad`) avec trois stratégies de remplissage des marges : `white`, `black` et `continuous` (extension des pixels de bordure).
* **Analyse en Composantes Principales (PCA) :**
  * Courbe de variance expliquée cumulée (*Scree plot* avec seuil à 90%).
  * Projection 2D de la base de données complète.
  * Reconstruction progressive d'une image en fonction du nombre $k$ de composantes principales et calcul de l'erreur $L_2$.
  * **Génération synthétique :** Création de nouveaux chiens aléatoires par tirage gaussien dans l'espace latent PCA de chaque race (`generer_et_afficher_toutes_races`).
* **Extraction de caractéristiques & Classification KNN :** Combinaison des projections PCA par classe (5 composantes par race) et du descripteur **HOG** (*Histogram of Oriented Gradients* calculé via les filtres de Sobel horizontaux et verticaux), suivie d'une standardisation (`StandardScaler`) et d'une classification par **KNN ($k=5$)**.

```bash
python Script01_PreprocessingExploration.py
```

#### 🔹 `script_SVC.py`
Ce script industrialise le flux de traitement en niveaux de gris à l'aide des outils avancés de `scikit-learn` :
* **Vérification du partitionnement :** Calcul de la **similarité cosinus** entre les distributions empiriques des jeux d'entraînement (`y_train`) et de test (`y_test`) pour garantir l'absence de biais d'échantillonnage.
* **Pipeline sans fuite de données (*Data Leakage*) :** Création de transformateurs personnalisés (`PCAInfoPreprocessing` et `EdgeInfoPreprocessing`) fusionnés via `FeatureUnion` dans un `Pipeline` complet (`MinMaxScaler` $\rightarrow$ `FeatureUnion(PCA + HOG)` $\rightarrow$ `StandardScaler` $\rightarrow$ `SVC`).
* **Optimisation d'hyperparamètres :** Recherche par grille (`GridSearchCV` à validation croisée) pour optimiser le nombre de composantes PCA, le coût `C` et le paramètre `gamma` du noyau RBF.
* **Comparaison Multiclasses (OvO vs OvR) :** Évaluation des stratégies *One-vs-One* et *One-vs-Rest* avec les hyperparamètres optimaux (comparaison de la précision, du temps d'entraînement et sauvegarde côte à côte des matrices de confusion dans `../folder_code2/comparaison_matrices_confusion.png`).

```bash
python script_SVC.py
```

---

### 2. Classification de races de chiens — Approche Couleur (RGB / HSV) & Étude du Padding

#### 🔹 `Script01_PreprocessingExploration_Orion.py`
Extension du premier script prenant en compte l'information chromatique (`color=True`) :
* **PCA 3D :** Visualisation du nuage de points projeté sur les 3 premiers vecteurs propres en trois dimensions.
* **Reconstruction PCA RGB :** Approximation d'images en couleur à travers différents paliers de composantes principales.
* **Descripteur HOG Couleur (`compute_hog_hsv`) :** Conversion des images de l'espace RGB vers l'espace **HSV** et extraction conjointe des histogrammes de gradients orientés sur le canal **H (Teinte)** et le canal **V (Luminosité)** afin de capturer à la fois la structure du pelage et sa couleur.

```bash
python Script01_PreprocessingExploration_Orion.py
```

#### 🔹 `script_2_Orion.py`
Pipeline SVM exploité sur les images en couleur et les descripteurs `compute_hog_hsv` :
* Génère le graphique comparatif des probabilités empiriques de chaque race entre la population globale, le jeu d'entraînement et le jeu de test.
* Compare les classifieurs SVM **OvO** et **OvR** sur les caractéristiques fusionnées (PCA couleur + HOG HSV).
* **Étude de l'impact du Padding :** Évalue systématiquement l'effet des 3 méthodes de remplissage (`white`, `black`, `continuous`) sur la précision finale en test des modèles OvO et OvR, et génère l'histogramme comparatif (`../folder_code2/padding_comparison.png`).

```bash
python script_2_Orion.py
```

---

### 3. Reconnaissance Faciale & Génération de Visages Synthétiques

#### 🔹 `nos_visages.py`
Script autonome appliquant la méthode des *Eigenfaces* sur les photos des membres du groupe (`./Datasets/att_faces`, images normalisées en `112x92`) :
1. **Reconnaissance faciale :** Réduction de dimension par PCA et classification par plus proche voisin (**1-NN**). Affiche les prédictions en vert/rouge (`figures/reconnaissance_faciale.png`) et génère la matrice de confusion (`figures/matrice_confusion_visages.png`).
2. **Génération de visages par IA (Espace Latent PCA) :**
   * Entraîne une PCA spécifique par personne et génère un visage synthétique inédit en échantillonnant un vecteur aléatoire gaussien calibré par l'écart-type des valeurs propres ($\sqrt{\lambda_i}$).
   * **Visage Chimère (`figures/visage_chimere.png`) :** Génère un visage hybride combinant les traits morphologiques de James, Orion et Raphaël.
3. **Approximation PCA (`figures/importante_features.png`) :** Reconstruit progressivement un visage de $k=1$ à $k=15$ composantes principales en affichant la décroissance de l'erreur $L_2$.

```bash
python nos_visages.py
```

---

## 👥 Auteurs
projet réalisé par :
- **James Duburcq** — [@jamesduburcq](https://github.com/jamesduburcq)
- **Raphaël Rideau** 
- **Orion Routhier** - [@OrionRouthier](https://github.com/OrionRouthier)
