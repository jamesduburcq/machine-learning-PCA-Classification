#### Import necessary libraries
# General libraries
import os
import random # Random number generators
import xml.etree.ElementTree as ET # For parsing XML like documents
# Numerical libraries
import numpy as np

# Visualisation
import matplotlib.pyplot as plt

# Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, hamming_loss
from sklearn.neighbors import KNeighborsClassifier

# Image processing
import skimage.io as io # image I/O routines
from skimage.transform import rescale, resize
from skimage.filters import sobel, sobel_h, sobel_v
import seaborn as sns

# We will add some macros to facilitate work later on. This is often good practice...
# Adapt here if you changed your workspace
##########################################
## Useful Macros & Parameters
##########################################
PATH_TO_DB = os.path.join('..', 'SmallDB')
IMG_DB = 'Images'
ANNOT_DB = 'Annotation'
IMG_FULL = os.path.join(PATH_TO_DB, IMG_DB)
FIGS = '.'

TARGET_SIZE = (128,128) 

os.makedirs('figures', exist_ok=True)

##########################################
## 1. Load Dataset
##########################################

def read_img(n_value, dog_breed, fname, TO_DB=IMG_FULL, color=True): 
    """Reads a single image using skimage.io."""
    folder_name = f"{n_value}-{dog_breed}"
    img_path = os.path.join(TO_DB, folder_name, fname)
    return io.imread(img_path, as_gray=not color)


def read_db(subset_dogs=None, TO_DB=IMG_FULL, color=True): 
    """Loads images, numeric labels, and human-readable label mapping."""
    label_names = dict()
    next_label = 0
    labels = []
    images = []

    if not os.path.exists(TO_DB):
        return None, None, None

    for subdir in sorted(os.listdir(TO_DB)):
        if '-' not in subdir:
            continue
        n_value, dog_breed = subdir.split('-', 1)
        
        if subset_dogs and dog_breed not in subset_dogs:
            continue
            
        label_names[next_label] = dog_breed
        nb_img = 0

        for img in os.listdir(os.path.join(TO_DB, subdir)):
            images.append(read_img(n_value, dog_breed, img, color=color))
            nb_img += 1
            
        labels.extend([next_label] * nb_img)
        next_label += 1

    return images, labels, label_names


##########################################
## 2. Exploratory Data Analysis
##########################################



def entropy(p):
    """
    ========================================================================
    TASK 1: SHANNON ENTROPY IMPLEMENTATION
    ========================================================================
    Computes Shannon Entropy of a discrete probability distribution.
    
    Formula: H(P) = -sum(p_i * log_e(p_i))
    
    Requirements:
    1. Convert 'p' to a floating-point numpy array.
    2. Normalize 'p' so that it sums to 1.0 (empirical probabilities).
    3. Filter out zero frequencies (p_i = 0) to avoid numerical log(0) errors.
    """
    ### STUDENT IMPLEMENTATION START ###
    
    p = np.array(p, dtype=np.float64)
    p = p / np.sum(p) 
    
    p = p[p > 0]
    
    entropy_value = -np.sum(p * np.log(p))
    
    ### STUDENT IMPLEMENTATION END ###
    return entropy_value

def entropy_breeds(labels):
    """Computes distribution entropy for evaluation of class balance."""
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts)

def plot_barplot(labels, label_names, fig_path = None):
    """Generates class balance bar plots."""
    l, v = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(l, v, color='#2e75b6', edgecolor='black', alpha=0.8)
    ax.set_xticks(l)
    ax.set_xticklabels([label_names[_l] for _l in l], rotation=15, ha='right')
    ax.set_ylabel("Sample Count")
    ax.set_title("SmallDB Class Balance Evaluation", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fig


##########################################
## 3. Localization and Preprocessing
##########################################

def plot_images(imgs, labels, label_names, nb_rows=3, nb_cols = 4, fig_path=None, rdm_indices=None):
    if not rdm_indices:
        # Random selection of indices from the database
        rdm_indices = random.sample(range(0, len(imgs)),nb_rows*nb_cols)
    fig, ax = plt.subplots(nb_rows, nb_cols, figsize=(10, 5))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                    top=0.9, wspace=0.4,hspace=0.4)
    for i in range(nb_rows):
        for j in range(nb_cols):
            ax[i][j].imshow(imgs[rdm_indices[i*nb_rows+j]], cmap='gray')
            ax[i][j].set_title(label_names[labels[rdm_indices[i*nb_rows+j]]])

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()

    return rdm_indices, fig

def extract_bounding_box(fname): 
    """Parses Pascal VOC XML files to retrieve object bounding coordinates."""
    tree = ET.parse(fname)
    root = tree.getroot()
    
    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)

    return xmin, xmax, ymin, ymax

def read_and_crop(n_value, dog_breed, fname, TO_DB=PATH_TO_DB, color=True): 
    """Reads source image alongside its XML bounding box data to crop target."""
    img = io.imread(os.path.join(TO_DB, IMG_DB, f"{n_value}-{dog_breed}", fname), as_gray=not color)
    base_name = os.path.splitext(fname)[0]
    bndbox_name = os.path.join(TO_DB, ANNOT_DB, f"{n_value}-{dog_breed}", base_name)
    
    xmin, xmax, ymin, ymax = extract_bounding_box(bndbox_name)
    dog = img[ymin:ymax+1, xmin:xmax+1]
    
    return img, dog

def read_and_crop_db(subset_dogs=None, TO_DB=PATH_TO_DB, color=True): 
    """Processes entire dataset, returning original and isolated target sub-arrays."""
    label_names = dict()
    next_label = 0
    labels = []
    images = []
    dogs = []

    img_root = os.path.join(TO_DB, IMG_DB)
    if not os.path.exists(img_root):
        return None, None, None, None

    for subdir in sorted(os.listdir(img_root)):
        if '-' not in subdir:
            continue
        n_value, dog_breed = subdir.split('-', 1)
        
        if subset_dogs and dog_breed not in subset_dogs:
            continue
            
        label_names[next_label] = dog_breed
        nb_img = 0
        
        for img_file in os.listdir(os.path.join(img_root, subdir)):
            img, dog = read_and_crop(n_value, dog_breed, img_file, TO_DB=TO_DB, color=color)
            images.append(img)
            dogs.append(dog)
            nb_img += 1
            
        labels.extend([next_label] * nb_img)
        next_label += 1

    return images, dogs, labels, label_names


##########################################
## 4. Dimensionality Reduction & Resizing
##########################################

def img_rescale(img, ratio): 
    if len(img.shape) == 3:
        return rescale(img, ratio, anti_aliasing=True, channel_axis=2)
    return rescale(img, ratio, anti_aliasing=True)

def img_resize(img, target_size): 
    return resize(img, target_size, anti_aliasing=True)

def resize_and_pad_maison(img, target_size=TARGET_SIZE, pad_type='continuous'):
    """
    ========================================================================
    TASK 2: ASPECT-PRESERVING SPATIAL SCALING
    ========================================================================
    Resizes an image preserving its original aspect ratio, centering it 
    on a fixed bounding canvas. At least three padding should be implemented
    * pad_type: Type of padding to use ('white' or 'black' or 'continuous').
        - 'white': Pads with maximum intensity (1.0 for normalized images).
        - 'black': Pads with minimum intensity (0.0 for normalized images).
        - 'continuous': Pads with the closest pixel value in the original image.
    Other potential padding strategies (e.g., reflection, edge replication, mean pixel values...) can be implemented as extensions.
    """
    h, w = img.shape[:2]
    height, width = target_size
    
    ### STUDENT IMPLEMENTATION START ###
    
    
    h_r = height / h
    w_r = width / w
    
    
    
    
    new_h = height
    new_w = width
    shape_out = (new_h, new_w, img.shape[2]) if img.ndim == 3 else (new_h, new_w)
    img_output = np.zeros(shape_out)
    
    if h >= w:
        img = img_rescale(img,h_r)
    else :
        img = img_rescale(img,w_r)
    
        
    
    if img.ndim == 2:    
        if h >= w:
            L_img = img.shape[1] 
            new_w = width
            marge = (new_w - L_img) // 2
            colonne_gauche = img[:,0]
            colonne_droite = img[:,-1]
            img_output[:, 0:marge] = colonne_gauche[:,np.newaxis]
            img_output[:,marge :marge + L_img] = img
            img_output[: , marge + L_img: ] = colonne_droite[:,np.newaxis]
            
            
        else :
            l_img = img.shape[0]
            new_h = height
            marge = (new_h - img.shape[0]) // 2
            ligne_haut = img[0,:]
            ligne_bas = img[-1,:]
            img_output[0:marge , :] = ligne_haut[np.newaxis,:]
            img_output[marge :marge + l_img ,:] = img
            img_output[marge + l_img :, :] = ligne_bas[np.newaxis,:]   
            
    elif img.ndim == 3:
        for c in range(img.ndim):
            if h >= w:
                L_img = img.shape[1] 
                new_w = height
                marge = (new_w - L_img) // 2
                colonne_gauche = img[:,0,c]
                colonne_droite = img[:,-1,c]
                img_output[:, 0:marge,c] = colonne_gauche[:,np.newaxis,c]
                img_output[:,marge :marge + L_img,c] = img[:,:,c]
                img_output[: , marge + L_img: ,c] = colonne_droite[:,np.newaxis,c]
                
                
            else :
                l_img = img.shape[0]
                new_h = width
                marge = (new_h - img.shape[0]) / 2
                ligne_haut = img[0,:,c]
                ligne_bas = img[-1,:,c]
                img_output[0:marge , :,c] = ligne_haut[np.newaxis,:,c] 
                img_output[marge :marge + l_img ,:,c] = img[:,:,c] 
                img_output[marge + l_img :, :,c] = ligne_bas[np.newaxis,:,c] 
        
    
    ### STUDENT IMPLEMENTATION END ###

    
    
    
        
    return img_output


def resize_and_pad(img, target_size=TARGET_SIZE, pad_type='continuous'):
    """
    ========================================================================
    TASK 2: ASPECT-PRESERVING SPATIAL SCALING
    ========================================================================
    Resizes an image preserving its original aspect ratio, centering it 
    on a fixed bounding canvas. At least three padding should be implemented
    * pad_type: Type of padding to use ('white' or 'black' or 'continuous').
        - 'white': Pads with maximum intensity (1.0 for normalized images).
        - 'black': Pads with minimum intensity (0.0 for normalized images).
        - 'continuous': Pads with the closest pixel value in the original image.
    """
    h, w = img.shape[:2]
    height, width = target_size
    
    ### STUDENT IMPLEMENTATION START ###
    
    
    scale = min(height / h, width / w) #On choisit le ratio minimum 
    
    
    img_scaled = img_rescale(img, scale)#fonction deja présente
    new_h, new_w = img_scaled.shape[:2]
    
    
    shape_out = (height, width, img.shape[2]) if img.ndim == 3 else (height, width)#On gère le noir et blanc ndim ==2 ou couleur ndim ==3
    
    
    if pad_type == 'white':
        img_output = np.ones(shape_out, dtype=img.dtype)   # 1.0 (ou 255 si non normalisé)
    else :
        img_output = np.zeros(shape_out, dtype=img.dtype)  #On laissera noir pour black et pour continuous on écrasera
    
    
    top = (height - new_h) // 2#Marge du haut
    bottom = top + new_h#Marge du bas
    left = (width - new_w) // 2#Marge de gauche
    right = left + new_w#Marge de droite
    
    img_output[top:bottom, left:right] = img_scaled
    
    if pad_type == 'continuous':
        
        if left > 0:
            img_output[top:bottom, :left] = img_scaled[:, :1]  
        if right < width:
            img_output[top:bottom, right:] = img_scaled[:, -1:]
            
        
        if top > 0:
            img_output[:top, :] = img_output[top:top+1, :]     
        if bottom < height:
            img_output[bottom:, :] = img_output[bottom-1:bottom, :] 
            
    ### STUDENT IMPLEMENTATION END ###
    
    return img_output


def get_resized_db(img_in, target_size=TARGET_SIZE, pad_type='white'):
    if len(img_in) == 0: return np.array([])
    has_channels = len(img_in[0].shape) >= 3
    shape_tuple = (len(img_in), target_size[0], target_size[1]) + ((img_in[0].shape[2],) if has_channels else ())
    output = np.zeros(shape_tuple)
    for idx, img in enumerate(img_in):
        output[idx] = resize_and_pad(img, target_size=target_size, pad_type=pad_type)
    return output

def convert_ndarrays2data_matrix(arr):
    nb_individuals = len(arr)
    dim = np.prod(arr.shape[1:])
    data_mtx = np.zeros((nb_individuals, dim))
    for i in range(nb_individuals): 
        data_mtx[i, :] = np.ravel(arr[i])
    return data_mtx


##########################################
## 5. Feature Engineering & Classification Pipeline
##########################################

def my_PCA(data, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca

def project_onto_PCA(n_components, pca_model, data):
    """
    ========================================================================
    TASK 3.1: CLASS-WISE PCA FEATURE EXTRACTION
    ========================================================================
    For a given PCA model, projects input data and extracts the top 'nb_components' principal components as features.
    """
    data_projected = pca_model.transform(data)
    
    return data_projected[:,0:n_components] # Remove this placeholder 

def visualize_var_pcs(pca, fig_path=None): 
    """
    ========================================================================
    TASK 3.2: PCA SCREE PLOT VISUALIZATION
    ========================================================================
    """
    exp_var_pca = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_pca)
    
    ### STUDENT IMPLEMENTATION START ###
    
    
    fig = plt.figure(figsize=(8, 4))
    
    
    axes_x = range(1, len(exp_var_pca) + 1)
    
    plt.bar(axes_x, exp_var_pca, label="Par composante")
    
    plt.plot(axes_x, cum_var, color="red", label="Total cumulé")
    
    plt.axhline(y=0.9, color="black", linestyle="--", label="Seu il 90%")
    
    plt.title("Graphique de la variance PCA")
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Quantité d'information (de 0 à 1)")
    plt.legend() 
    plt.show()
    
    if fig_path: 
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fig

def plot_whole_db_on_2d(pca, data_mtx, fig_path=None):
    """
    ========================================================================
    TASK 3.3: PCA PROJECTION SCATTER PLOT
    ========================================================================
    Projects the entire dataset onto the first two principal components and visualizes it as a scatter plot, colored by class labels.
    """
    projected_data = project_onto_PCA(2, pca, data_mtx)

    fig = sns.scatterplot(x=projected_data[:, 0], y=projected_data[: , 1] , hue = labels, palette="deep" , alpha = 0.7)
    plt.show()
    
    return fig 

def display_pca_approx(img, pca, target_size=TARGET_SIZE, fig_path=None): 
    """
    ========================================================================
    TASK 4: PCA RECONSTRUCTION APPROXIMATION
    ========================================================================
    For a given input image, reconstructs approximations using increasing numbers of principal components (k) and visualizes the original and reconstructed images side by side, along with their respective L2 reconstruction errors.
    """
    img_flat = img.reshape(1, -1)
    original_image = img_flat.reshape(target_size[0], target_size[1])  
    

    fig = plt.figure(figsize=(15,15))
    plt.subplot(4,4,1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original image")
    plt.axis('off')

    

    k = [20,50,100,120,150,200,250,300,400,500,700,900,1000,1100,1200]
    img_reduit = pca.transform(img_flat)
    
    i = 2
    for k_i in k:
        img_k = img_reduit[:,:k_i]
        img_reconstruit_flat = (np.dot(img_k, pca.components_[:k_i, :]) + pca.mean_)
        img_reconstruit = img_reconstruit_flat.reshape(target_size[0], target_size[1])
        l2_erreur = np.linalg.norm(original_image - img_reconstruit)
        plt.subplot(4, 4, i)
        plt.imshow(img_reconstruit, cmap="gray")
        plt.title(f"k = {k_i}\nL2 Error: {l2_erreur:.2f}", fontsize=10)
        plt.axis("off")
        i+=1
    plt.tight_layout()
    
            

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    
    
    plt.show()
    return fig 

def creation_chien(pca, target_size = TARGET_SIZE, seed = 12):
    
    exp_var_pca = pca.explained_variance_
    n = pca.n_components_
    rand_pca = np.random.normal(loc = 0, scale = exp_var_pca, size=n)
    img_flat_gen = (np.dot(rand_pca.reshape(1,-1), pca.components_)) + pca.mean_
    img_2d = img_flat_gen.reshape(target_size[0], target_size[1])
    plt.imshow(img_2d,cmap = 'gray')
    return img_2d

def generer_et_afficher_toutes_races(data_train, y_train, label_names, n_components=15, target_size=TARGET_SIZE):
    """
    Entraîne une PCA par classe et génère un chien synthétique aléatoire 
    pour chaque race sous forme de grille comparative.
    """
    classes = sorted(list(label_names.keys()))
    num_classes = len(classes)
    
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
    
    if num_classes == 1:
        axes = [axes]

    for idx, l in enumerate(classes):
        class_subset = data_train[y_train == l]
        
        components_to_use = min(n_components, len(class_subset) - 1)
        
        pca_class = PCA(n_components=components_to_use, random_state=42)
        pca_class.fit(class_subset)
        
        
        exp_var_pca = pca_class.explained_variance_
        rand_pca = np.random.normal(loc=0, scale=np.sqrt(exp_var_pca), size=components_to_use)
        
        img_flat_gen = (np.dot(rand_pca.reshape(1, -1), pca_class.components_)) + pca_class.mean_
        img_2d = img_flat_gen.reshape(target_size[0], target_size[1])
        
        axes[idx].imshow(img_2d, cmap='gray')
        axes[idx].set_title(f"Génération : {label_names[l]}", fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()
    return fig
    
    
    

def compute_pca_features(pcas, input_data, nb_components=5):
    out = np.zeros((len(input_data), 0))
    for pca in pcas:
        proj = pca.transform(input_data)
        features = proj[:, :min(proj.shape[1], nb_components)]
        out = np.hstack((out, features))
    return out

def compute_hog(image, nb_height_cells=4, nb_width_cells=4, nb_bins=8):
    """
    ========================================================================
    TASK 5: CLASSICAL HISTOGRAM OF ORIENTED GRADIENTS (HOG)
    ========================================================================
    Decomposes an image into local grid patches, evaluates edge orientations, 
    and accumulates magnitudes within discrete orientation channels.
    """
    h_edge = sobel_h(image)
    v_edge = sobel_v(image)
    magnitude = np.sqrt(h_edge**2 + v_edge**2)

    orientation = np.arctan2(v_edge, h_edge)
    orientation[orientation < 0] += np.pi # Constrain range within [0, \pi)

    H, W = image.shape[:2]
    cell_h = H // nb_height_cells
    cell_w = W // nb_width_cells
    bin_width = np.pi / nb_bins

    output = np.zeros((nb_height_cells, nb_width_cells, nb_bins))

    ### STUDENT IMPLEMENTATION START ###
    
    # TODO: Complete the nested loops below to build your spatial histogram maps
    for i in range(nb_height_cells):
        for j in range(nb_width_cells):
            for p_i in range (cell_h):
                for p_j in range(cell_w):
                    angle = orientation[i * cell_h + p_i, j * cell_w + p_j]
                    bin_idx = int(angle // bin_width)
                    if bin_idx >= nb_bins:
                        bin_idx = nb_bins - 1
                    output[i , j , bin_idx ] += magnitude[i * cell_h + p_i, j * cell_w + p_j]

    return output.reshape(-1)

def visualize_var_pcs_classe(pca, nom_classe, fig_path=None): 
    exp_var_pca = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_pca)
    
    fig = plt.figure(figsize=(8, 4))
    axes_x = range(1, len(exp_var_pca) + 1)
    
    # Graphique en barres + courbe cumulative
    plt.bar(axes_x, exp_var_pca, label="Par composante")
    plt.plot(axes_x, cum_var, color="red", label="Total cumulé")
    plt.axhline(y=0.9, color="black", linestyle="--", label="Seuil 90%")
    
    # Personnalisation du titre avec le nom de la race
    plt.title(f"Graphique de la variance PCA - Race : {nom_classe}")
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Quantité d'information (de 0 à 1)")
    plt.legend() 
    
    if fig_path: 
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    return fig



# =========================================================================
if __name__ == '__main__':
    print("Step 1: Loading Dataset")
    
    bw_imgs, bw_dogs, labels, label_names = read_and_crop_db(color=False)
    img_brute = bw_dogs[0] 

    # --- MODIFICATION DEMANDÉE : AFFICHAGE SÉQUENTIEL SANS SUBPLOTS ---
    # 1. Image originale seule
    plt.figure(figsize=(4, 4))
    plt.imshow(img_brute, cmap='gray')
    plt.title(f"Originale ({img_brute.shape[0]}x{img_brute.shape[1]})")
    plt.axis('off')
    plt.show()

    # 2. Les 3 paddings affichés l'un après l'autre
    for padding_mode in ['white', 'black', 'continuous']:
        img_calculee = resize_and_pad(img_brute, pad_type=padding_mode)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(img_calculee, cmap='gray')
        plt.title(f"Après resize_and_pad [Mode: {padding_mode}] {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
        plt.axis('off')
        plt.show()
    # -----------------------------------------------------------------
    
    if bw_imgs is None:
        print(" SmallDB not found.")
        label_names = {0: 'Chihuahua', 1: 'Pug', 2: 'Malamute', 3: 'Beagle'}
        labels = [0]*152 + [1]*250 + [2]*300 + [3]*300
        
        np.random.seed(42)
        sample_img = np.zeros((70, 70))
        sample_img[20:55, 15:55] = 0.6
        bw_imgs = [sample_img] * len(labels)
        bw_dogs = [sample_img[10:60, 10:60]] * len(labels)
        
        data_mtx = np.random.rand(len(labels), TARGET_SIZE[0] * TARGET_SIZE[1]) * 0.4
        for idx, lbl in enumerate(labels):
            data_mtx[idx, 400:1500] += lbl * 0.1
    else:
        print(" Real dataset successfully resolved.")
        resized_dogs = get_resized_db(bw_dogs, target_size=TARGET_SIZE, pad_type='white')
        data_mtx = convert_ndarrays2data_matrix(resized_dogs)

    # Calculate Shannon Entropy
    db_entropy = entropy_breeds(labels)
    print(f"   - Dataset Balance Shannon Entropy: {db_entropy:.4f}")
    plot_barplot(labels, label_names)
    plt.close()

    print("\n Step 2: Saving Localization & Bounding Box Crops")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.imshow(bw_imgs[0], cmap='gray')
    ax0.set_title("Original Asset Image Frame", fontsize=9)
    ax0.axis('off')
    ax1.imshow(bw_dogs[0], cmap='gray')
    ax1.set_title("XML Isolated Target Region", fontsize=9)
    ax1.axis('off')
    plt.savefig('figures/bounding_box_crop.png', bbox_inches='tight', dpi=150)
    plt.close()

    print("\n Step 3: Saving Aspect-Preserving Scaling Validations")
    asymmetric_test = np.zeros((20, 80))
    asymmetric_test[4:16, 10:70] = 0.8
    stretched_test = resize(asymmetric_test, TARGET_SIZE, anti_aliasing=True)
    padded_test = resize_and_pad(asymmetric_test, target_size=TARGET_SIZE, pad_type='white')
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.imshow(stretched_test, cmap='gray')
    ax0.set_title("Stretched Distorted Scaling", fontsize=9)
    ax0.axis('off')
    ax1.imshow(padded_test, cmap='gray')
    ax1.set_title("Aspect-Preserved Padding Scaling", fontsize=9)
    ax1.axis('off')
    plt.savefig(os.path.join(FIGS, 'resize_vs_pad_whiteband.png'), bbox_inches='tight', dpi=150)
    plt.close()

    print("\n Step 4: Stratifying Splits and Training Class PCA Engines")
    data_train, data_test, y_train, y_test = train_test_split(
        data_mtx, labels, test_size=0.25, stratify=labels, random_state=42
    )
    y_train, y_test = np.array(y_train), np.array(y_test)

    pca_global = my_PCA(data_train)
    
    
    visualize_var_pcs(pca_global, fig_path=os.path.join(FIGS,'pca_reconstruction.png'))

    plot_whole_db_on_2d(pca_global, data_mtx, fig_path=os.path.join(FIGS, "pca_scatter_plots.png")) # Note that this matrix includes both test and training sets
    
    display_pca_approx(data_mtx[0], pca_global, fig_path=os.path.join(FIGS, "pca_approximatrion.png")) 

    pca_per_class = []
    nb_pcs_per_class = 5
    for l in sorted(list(label_names.keys())):
        class_subset = data_train[y_train == l]
        pca_per_class.append(my_PCA(class_subset, n_components=nb_pcs_per_class))
        
    image_genere = creation_chien(pca_per_class[1])

    print("\n Step 5: Extracting Localized Structural Features")
    target_canvas = data_train[0].reshape(TARGET_SIZE)
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    axes[0, 0].imshow(target_canvas, cmap='gray')
    axes[0, 0].set_title("Target Image", fontsize=8)
    axes[0, 1].imshow(sobel(target_canvas), cmap='gray')
    axes[0, 1].set_title("Sobel Magnitude", fontsize=8)
    axes[1, 0].imshow(sobel_v(target_canvas), cmap='gray')
    axes[1, 0].set_title("Vertical $g_y$", fontsize=8)
    axes[1, 1].imshow(sobel_h(target_canvas), cmap='gray')
    axes[1, 1].set_title("Horizontal $g_x$", fontsize=8)
    for ax in axes.ravel(): ax.axis('off')
    plt.savefig(os.path.join(FIGS, 'sobel_hog.png'), bbox_inches='tight', dpi=150)
    plt.close()

    X_train_pca = compute_pca_features(pca_per_class, data_train, nb_components=nb_pcs_per_class)
    X_train_hog = np.array([compute_hog(img.reshape(TARGET_SIZE)) for img in data_train])
    X_train_combined = np.hstack((X_train_pca, X_train_hog))

    X_test_pca = compute_pca_features(pca_per_class, data_test, nb_components=nb_pcs_per_class)
    X_test_hog = np.array([compute_hog(img.reshape(TARGET_SIZE)) for img in data_test])
    X_test_combined = np.hstack((X_test_pca, X_test_hog))

    print("\n Step 6: Standardizing Features & Optimizing Classifiers")
    
    # =========================================================================
    # TASK 4: ANALYSIS 
    # =========================================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    knn = KNeighborsClassifier(n_neighbors=5)
    
    knn.fit(X_train_scaled, y_train)
    train_err = hamming_loss(y_train, knn.predict(X_train_scaled))
    test_err = hamming_loss(y_test, knn.predict(X_test_scaled))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(['Training Set Error', 'Testing Generalization Loss'], [train_err, test_err], color=['#a6a6a6', '#c00000'], height=0.4, edgecolor='black')
    ax.set_xlim(0, max(train_err, test_err) + 0.1)
    ax.set_title("Model Generalization Diagnostic Gap", fontweight='bold')
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width()*100:.1f}%', va='center', fontweight='bold')
    plt.savefig(os.path.join(FIGS, 'train_test_error.png'), bbox_inches='tight', dpi=150)
    plt.close()

    print("\n" + "="*50)
    print("DIAGNOSTIC RESULTS SUMMARY")
    print("="*50)
    print(f" Training Error Rate (Hamming Loss): {train_err*100:.2f}%")
    print(f" Testing Error Rate (Generalization Loss):  {test_err*100:.2f}%")
    print("="*50)
    print(f"All figures successfully saved inside the '{FIGS}/' folder.")
    
    print("\n Step 7: Generating Confusion Matrix")
    y_pred = knn.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred)
    
    target_names = [label_names[i] for i in sorted(label_names.keys())]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=15)
    ax.set_title("Matrice de Confusion par Race de Chien", fontweight='bold', pad=15)
    
    plt.show()
    plt.savefig(os.path.join(FIGS, 'figures/matrice_confusion.png'), bbox_inches='tight', dpi=150)
    
    print("   - Confusion matrix successfully saved as 'figures/matrice_confusion.png'")
    print("\n -> Génération d'un spécimen aléatoire pour chaque race...")
    fig_races = generer_et_afficher_toutes_races(data_train, y_train, label_names, n_components=15)
    classe_cible = 0
    nom_classe = label_names[classe_cible] # Récupère le nom de la race (ex: 'Chihuahua')
    
    
    class_subset = data_train[y_train == classe_cible]
    
    
    pca_classe = my_PCA(class_subset)
    visualize_var_pcs_classe(pca_classe, nom_classe)
    
         