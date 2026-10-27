import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================================================
# CONFIGURATION & CRÉATION DU DOSSIER DE SAUVEGARDE
# =========================================================================
TARGET_SIZE = (112, 92)  # Hauteur: 112, Largeur: 92
BASE_PATH = "./Datasets/att_faces"
FIGS = '.'

# FIX : On crée le dossier de sauvegarde dès le début pour éviter les FileNotFoundError
output_dir = os.path.join(FIGS, 'figures')
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(BASE_PATH):
    raise FileNotFoundError(f"Le dossier {BASE_PATH} est introuvable. Vérifie ton chemin.")

NOMS_DOSSIERS = sorted([
    d for d in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, d))
])

# Mapping pour faire le lien entre les chiffres et vos noms
LABEL_TO_NAME = {i: name for i, name in enumerate(NOMS_DOSSIERS)}

# =========================================================================
# FONCTIONS DE PRÉTRAITEMENT
# =========================================================================
def img_rescale(img, ratio): 
    if len(img.shape) == 3:
        return rescale(img, ratio, anti_aliasing=True, channel_axis=2)
    return rescale(img, ratio, anti_aliasing=True)

def resize_and_pad(img, target_size=TARGET_SIZE, pad_type='white'):
    h, w = img.shape[:2]
    height, width = target_size
    
    scale = min(height / h, width / w) 
    img_scaled = img_rescale(img, scale)
    new_h, new_w = img_scaled.shape[:2]
    
    shape_out = (height, width, img.shape[2]) if img.ndim == 3 else (height, width)
    if pad_type == 'white':
        img_output = np.ones(shape_out, dtype=np.float32)
    else:
        img_output = np.zeros(shape_out, dtype=np.float32)
    
    top = (height - new_h) // 2
    bottom = top + new_h
    left = (width - new_w) // 2
    right = left + new_w
    
    img_output[top:bottom, left:right] = img_scaled
    return img_output

def get_resized_db(img_in, target_size=TARGET_SIZE, pad_type='white'):
    if len(img_in) == 0: return np.array([])
    has_channels = len(img_in[0].shape) >= 3
    shape_tuple = (len(img_in), target_size[0], target_size[1]) + ((img_in[0].shape[2],) if has_channels else ())
    output = np.zeros(shape_tuple)
    for idx, img in enumerate(img_in):
        output[idx] = resize_and_pad(img, target_size=target_size, pad_type=pad_type)
    return output

# =========================================================================
# CHARGEMENT DES DONNÉES
# =========================================================================
def read_custom_db(base_path, folder_names, color=False):
    images = []
    labels = []
    for idx, folder in enumerate(folder_names):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            print(f"Lecture des images de {folder}...")
            for file in sorted(os.listdir(folder_path)):
                if file.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg')):
                    img = io.imread(os.path.join(folder_path, file), as_gray=not color)
                    images.append(img)
                    labels.append(idx) 
    return images, np.array(labels)

# =========================================================================
# TASK 4: PCA RECONSTRUCTION APPROXIMATION
# =========================================================================
def display_pca_approx_james(X_data, labels, label_to_name, target_name="James", target_size=TARGET_SIZE, fig_path=None): 
    idx_james = None
    for idx, name in label_to_name.items():
        if target_name.lower() in name.lower():
            idx_james = idx
            break
    
    if idx_james is None:
        raise ValueError(f"La classe '{target_name}' n'a pas été trouvée.")
    
    X_james = X_data[labels == idx_james]
    if len(X_james) == 0:
        raise ValueError(f"Aucune image trouvée pour : {target_name}")
        
    n_samples = len(X_james)
    n_components = min(15, n_samples - 1) 
    
    pca_james = PCA(n_components=n_components, random_state=42)
    pca_james.fit(X_james)
    
    img_flat = X_james[0].reshape(1, -1)
    original_image = img_flat.reshape(target_size)  

    fig = plt.figure(figsize=(15, 15))
    plt.subplot(4, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(f"Original ({target_name})")
    plt.axis('off')

    k_values = list(range(1, n_components + 1))
    img_reduit = pca_james.transform(img_flat)
    
    for i, k_i in enumerate(k_values):
        if i >= 15: break  
        
        img_k = img_reduit[:, :k_i]
        img_reconstruit_flat = (np.dot(img_k, pca_james.components_[:k_i, :]) + pca_james.mean_)
        img_reconstruit = img_reconstruit_flat.reshape(target_size)
        
        l2_erreur = np.linalg.norm(original_image - img_reconstruit)
        
        plt.subplot(4, 4, i + 2) 
        plt.imshow(img_reconstruit, cmap="gray")
        plt.title(f"k = {k_i}\nL2 Error: {l2_erreur:.2f}", fontsize=10)
        plt.axis("off")
        
    plt.tight_layout()

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.savefig(os.path.join(output_dir, "importante_features.png"), bbox_inches='tight', dpi=150)
    plt.show()
    return fig

# Execution du chargement
raw_images, labels = read_custom_db(BASE_PATH, NOMS_DOSSIERS)
resized_images = get_resized_db(raw_images, target_size=TARGET_SIZE, pad_type='white')
data_mtx = resized_images.reshape(len(resized_images), -1)

X_train, X_test, y_train, y_test = train_test_split(
    data_mtx, labels, test_size=0.20, stratify=labels, random_state=42
)

# =========================================================================
# PARTIE 1 : RECONNAISSANCE FACIALE (PREDICTION)
# =========================================================================
print("\n--- Entraînement du modèle de reconnaissance ---")
n_comp_rec = min(15, len(X_train) - 1)
pca_rec = PCA(n_components=n_comp_rec)

X_train_pca = pca_rec.fit_transform(X_train)
X_test_pca = pca_rec.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train_pca, y_train)
predictions = classifier.predict(X_test_pca)

# FIX : On crée une belle grille de sous-graphiques côte à côte
plt.figure(figsize=(12, 5))
nb_affichage = min(3, len(X_test))
for i in range(nb_affichage):
    plt.subplot(1, nb_affichage, i + 1) # FIX : Ajout des subplots
    plt.imshow(X_test[i].reshape(TARGET_SIZE), cmap='gray')
    
    vrai_nom = LABEL_TO_NAME[y_test[i]]
    nom_predit = LABEL_TO_NAME[predictions[i]]
    
    couleur = 'green' if vrai_nom == nom_predit else 'red'
    plt.title(f"Vrai : {vrai_nom}\nPrédit : {nom_predit}", color=couleur, fontsize=12)
    plt.axis('off')

plt.suptitle("Test de Reconnaissance Faciale", fontsize=16, fontweight='bold')
# FIX : Sauvegarde à l'extérieur de la boucle AVANT le plt.show()
plt.savefig(os.path.join(output_dir, "reconnaissance_faciale.png"), bbox_inches='tight', dpi=150)
plt.show()

# =========================================================================
# PARTIE 2 : GÉNÉRATION DE VISAGES SYNTHÉTIQUES
# =========================================================================
print("\n--- Génération des visages synthétiques ---")

def generer_visage_aleatoire(X_person, target_size=TARGET_SIZE, tirage='gaussien'):
    n_samples = X_person.shape[0]
    if n_samples <= 1:
        return X_person[0].reshape(target_size)
        
    n_components = min(10, n_samples - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_person)
    
    if tirage == 'gaussien':
        vecteur_aleatoire = np.random.randn(n_components)
    elif tirage == 'uniform':
        vecteur_aleatoire = np.random.rand(n_components)
    else:
        return np.zeros(target_size)
        
    ecarts_types = np.sqrt(pca.explained_variance_)
    vecteur_latent_calibre = vecteur_aleatoire * ecarts_types
    
    image_plate = pca.inverse_transform(vecteur_latent_calibre.reshape(1, -1))
    return image_plate.reshape(target_size)

# FIX : Création d'une figure propre par personne à l'intérieur de la boucle
for idx, name in LABEL_TO_NAME.items():
    plt.figure(figsize=(4, 5)) # FIX : Initialisation ici pour éviter les écrasements
    X_person = X_train[y_train == idx]
    
    visage_genere = generer_visage_aleatoire(X_person, target_size=TARGET_SIZE)
    plt.imshow(visage_genere, cmap='gray')
    plt.title(f"Visage IA : {name}", fontsize=12)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"visage_{name}.png"), bbox_inches='tight', dpi=150)
    plt.show()

# Visage chimère
plt.figure(figsize=(5, 5))
visage_chimere = generer_visage_aleatoire(X_train, target_size=TARGET_SIZE)
plt.imshow(visage_chimere, cmap='gray')
plt.title("Visage combiné de James, Orion et Raphael")
plt.axis('off') # FIX : Placé AVANT la sauvegarde pour enlever les bordures noires
plt.savefig(os.path.join(output_dir, 'visage_chimere.png'), bbox_inches='tight', dpi=150)
plt.show()

# =========================================================================
# PARTIE 3 : MATRICE DE CONFUSION
# =========================================================================
cm = confusion_matrix(y_test, predictions)
target_names = NOMS_DOSSIERS

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=15)
ax.set_title("Matrice de Confusion des visages", fontweight='bold', pad=15)

plt.savefig(os.path.join(output_dir, 'matrice_confusion_visages.png'), bbox_inches='tight', dpi=150)
plt.show()
print("   - All charts successfully saved in the 'figures/' directory.")

# =========================================================================
# EXECUTION DE LA RECONSTRUCTION DE JAMES
# =========================================================================
display_pca_approx_james(data_mtx, labels, LABEL_TO_NAME, target_name="James")