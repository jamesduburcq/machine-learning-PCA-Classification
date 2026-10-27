#### Import necessary libraries
# General libraries
import os
import random # Random number generators
import xml.etree.ElementTree as ET # For parsing XML like documents
# Numerical libraries
import numpy as np
from skimage import color
import skimage.io as io

# Visualisation
import matplotlib.pyplot as plt

# Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, hamming_loss
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
# Image processing
import skimage.io as io # image I/O routines
from skimage.transform import rescale, resize
from skimage.filters import sobel, sobel_h, sobel_v

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
        img_output = np.ones(shape_out, dtype=img_scaled.dtype)
    else :
        img_output = np.zeros(shape_out, dtype=img_scaled.dtype)
   
   
    #Calcul des marges
    top = (height - new_h) // 2#Marge du haut
    bottom = top + new_h#Marge du bas
    left = (width - new_w) // 2#Marge de gauche
    right = left + new_w#Marge de droite
   
    # Incrustation de l'image redimensionnée au centre
    img_output[top:bottom, left:right] = img_scaled
   
    if pad_type == 'continuous':
        # Remplissage horizontal (gauche et droite)
        if left > 0:
            img_output[top:bottom, :left] = img_scaled[:, :1]  # Duplication de la première colonne
        if right < width:
            img_output[top:bottom, right:] = img_scaled[:, -1:] # Duplication de la dernière colonne
           
        # Remplissage vertical (haut et bas) sur toute la largeur
        # En faisant cela après, les coins se remplissent automatiquement de manière cohérente !
        if top > 0:
            img_output[:top, :] = img_output[top:top+1, :]     # Duplication de la première ligne
        if bottom < height:
            img_output[bottom:, :] = img_output[bottom-1:bottom, :] # Duplication de la dernière ligne
           
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
    ### STUDENT IMPLEMENTATION START ###
   
   
    return pca_model.transform(data)[:,0:n_components] # Remove this placeholder

def visualize_var_pcs(pca, fig_path=None):
    """
    ========================================================================
    TASK 3.2: PCA SCREE PLOT VISUALIZATION
    ========================================================================
    Plots the individual and cumulative explained variance ratios for each principal component, with a reference line at 90% cumulative variance.
    """
    exp_var_pca = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_pca)
    #
    n=range(1,len(exp_var_pca)+1)
    plt.figure()
    plt.plot(n, cum_var, color='red')
    plt.bar(n,exp_var_pca)
    plt.axhline(y=0.9)
    plt.title('variance expliquée en fonction de variance cumulée')
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
    projected_data = project_onto_PCA(3, pca, data_mtx)
   
    fig = plt.figure(figsize=(10, 8))
   
    # 2. Ajout explicite d'un axe en 3 dimensions
    ax = fig.add_subplot(111, projection='3d')
   
    # 3. Création du nuage de points 3D
    nuage = ax.scatter(
        projected_data[:, 0],  # Axe X : Composante 1
        projected_data[:, 1],  # Axe Y : Composante 2
        projected_data[:, 2],  # Axe Z : Composante 3
        c=labels,              # La couleur dépend de la classe du chien
        cmap='viridis',        # L'équivalent de palette="deep" pour matplotlib
        s=40,                  # Taille des points
        alpha=0.7              # Transparence
    )
   
    # 4. Ajout des labels sur les axes (Note l'utilisation de ax.set_...)
    ax.set_xlabel("Premier vecteur principal")
    ax.set_ylabel("Deuxième vecteur principal")
    ax.set_zlabel("Troisième vecteur principal") # Corrigé : 3ème vecteur pour l'axe Z
    ax.set_title("Projection PCA en 3D", fontweight='bold')
   
    # 5. Ajout de la légende basée sur les couleurs générées
    legend = ax.legend(*nuage.legend_elements(), title="Classes", loc="best")
    ax.add_artist(legend)
   
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
    original_image = img_flat.reshape(target_size[0], target_size[1],3)  
   

    fig = plt.figure(figsize=(15,15))
    plt.subplot(4,4,1)
   
    plt.imshow(original_image)
    plt.title("Original image")
    plt.axis('off')

   

    k = [20,50,100,120,150,200,250,300,400,500,700,900,1000,1100,1200]
    img_reduit = pca.transform(img_flat)
   
    i = 2
    for k_i in k:
        img_k = img_reduit[:,:k_i]
        img_reconstruit_flat = (np.dot(img_k, pca.components_[:k_i, :]) + pca.mean_)
        img_reconstruit = img_reconstruit_flat.reshape(target_size[0], target_size[1],3)
       
        img_reconstruit_safe = np.clip(img_reconstruit, 0, 1)
       
        l2_erreur = np.linalg.norm(original_image - img_reconstruit)
        plt.subplot(4, 4, i)
        plt.imshow(img_reconstruit_safe)
        plt.title(f"k = {k_i}\nL2 Error: {l2_erreur:.2f}", fontsize=10)
        plt.axis("off")
        i+=1
    plt.tight_layout()
   
           

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
   
   
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

    for i in range(nb_height_cells):
        for j in range(nb_width_cells):
            for k in range(cell_h) :
               for l in range(cell_w) :
                   angle = orientation[i*cell_h+k,j*cell_w+l]
                   bins_ind=int(angle//bin_width)
                   if bins_ind==nb_bins :
                       bins_ind -=1                  
                   output[i,j,bins_ind] += magnitude[i*cell_h+k,j*cell_w+l]        



    return output.reshape(-1)
     

def compute_hog_hsv(image, nb_height_cells=4, nb_width_cells=4, nb_bins=8):
        """
        ========================================================================
        TASK 5: CLASSICAL HISTOGRAM OF ORIENTED GRADIENTS (HOG)
        ========================================================================
        Decomposes an image into local grid patches, evaluates edge orientations,
        and accumulates magnitudes within discrete orientation channels.
        """
       
        img_hsv = color.rgb2hsv(image)
        img_h=img_hsv[:,:,0]
        img_v=img_hsv[:,:,2]
       
        h_edge_h = sobel_h(img_h)
        v_edge_h = sobel_v(img_h)
        h_edge_v = sobel_h(img_v)
        v_edge_v = sobel_v(img_v)
        magnitude_h = np.sqrt(h_edge_h**2 + v_edge_h**2)
        magnitude_v = np.sqrt(h_edge_v**2 + v_edge_v**2)

        orientation_h = np.arctan2(v_edge_h, h_edge_h)
        orientation_v = np.arctan2(v_edge_v, h_edge_v)
        orientation_h[orientation_h < 0] += np.pi # Constrain range within [0, \pi)
        orientation_v[orientation_v < 0] += np.pi # Constrain range within [0, \pi)

        H, W = image.shape[:2]
        cell_h = H // nb_height_cells
        cell_w = W // nb_width_cells
        bin_width = np.pi / nb_bins

        output_h = np.zeros((nb_height_cells, nb_width_cells, nb_bins))
        output_v = np.zeros((nb_height_cells, nb_width_cells, nb_bins))

        for i in range(nb_height_cells):
            for j in range(nb_width_cells):
                for k in range(cell_h) :
                   for l in range(cell_w) :
                       angle_h = orientation_h[i*cell_h+k,j*cell_w+l]
                       angle_v = orientation_v[i*cell_h+k,j*cell_w+l]
                       bins_ind_h=int(angle_h//bin_width)
                       bins_ind_v=int(angle_v//bin_width)
                       if bins_ind_h==nb_bins :
                           bins_ind_h-=1      
                   
                       if bins_ind_v==nb_bins :
                               bins_ind_v-=1
               
                       output_h[i,j,bins_ind_h] += magnitude_h[i*cell_h+k,j*cell_w+l]    
                       output_v[i,j,bins_ind_v] += magnitude_v[i*cell_h+k,j*cell_w+l]

        output = np.concatenate((output_h.reshape(-1),output_v.reshape(-1)))
        return output.reshape(-1)
         

# =========================================================================
if __name__ == '__main__':
    print("Step 1: Loading Dataset")
   
    bw_imgs, bw_dogs, labels, label_names = read_and_crop_db(color=True) # pour qu'on considere les images en coulerus
    img_brute = bw_dogs[1]
    # 2. On applique ta fonction avec le padding de ton choix ('continuous', 'white', 'black')
    img_calculee = resize_and_pad(img_brute, pad_type='continuous')
    # 3. On crée la figure pour comparer
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
   
    ax0.imshow(img_brute, cmap='gray')
    ax0.set_title(f"Originale ({img_brute.shape[0]}x{img_brute.shape[1]})")
    ax0.axis('off')
   
    ax1.imshow(img_calculee, cmap='gray')
    ax1.set_title(f"Après resize_and_pad ({TARGET_SIZE[0]} x {TARGET_SIZE[1]})")
    ax1.axis('off')
   
    # 4. Affichage
    plt.show()
   
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

    # Global PCA Scree Plot mapping
    pca_global = my_PCA(data_train)
    visualize_var_pcs(pca_global, fig_path=os.path.join(FIGS,'pca_reconstruction.png'))

    # PCA database visualisation scatter plot
    plot_whole_db_on_2d(pca_global, data_mtx, fig_path=os.path.join(FIGS, "pca_scatter_plots.png")) # Note that this matrix includes both test and training sets
   
    # PCA Approximation plots
    display_pca_approx(data_mtx[0], pca_global, fig_path=os.path.join(FIGS, "pca_approximatrion.png"))

    # PCA Feature engineering: Per-class PCA modeling loops
    pca_per_class = []
    nb_pcs_per_class = 5
    for l in sorted(list(label_names.keys())):
        class_subset = data_train[y_train == l]
        pca_per_class.append(my_PCA(class_subset, n_components=nb_pcs_per_class))

    print("\n Step 5: Extracting Localized Structural Features")
    print("\n Step 5: Extracting Localized Structural Features")
   
    # 1. On récupère l'image en RGB
    target_canvas_3d = data_train[0].reshape(TARGET_SIZE[0], TARGET_SIZE[1], 3)
   
    # 2. On convertit localement en HSV pour extraire proprement le canal V
    target_canvas_hsv = color.rgb2hsv(target_canvas_3d)
    canal_v_sobel = target_canvas_hsv[:, :, 2]
   
    # 3. Création de la figure
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
   
    # Case [0,0] : L'image originale (on l'affiche directement en RGB)
    axes[0, 0].imshow(target_canvas_3d)
    axes[0, 0].set_title("Target Image", fontsize=8)
       
    # Case [0,1] : La magnitude du gradient (Sobel complet)
    axes[0, 1].imshow(sobel(canal_v_sobel), cmap='gray')
    axes[0, 1].set_title("Sobel Magnitude", fontsize=8)
   
    # Case [1,0] : Le gradient vertical
    axes[1, 0].imshow(sobel_v(canal_v_sobel), cmap='gray')
    axes[1, 0].set_title("Vertical $g_y$", fontsize=8)
   
    # Case [1,1] : Le gradient horizontal
    axes[1, 1].imshow(sobel_h(canal_v_sobel), cmap='gray')
    axes[1, 1].set_title("Horizontal $g_x$", fontsize=8)
   
    for ax in axes.ravel():
        ax.axis('off')
       
    plt.savefig(os.path.join(FIGS, 'sobel_hog.png'), bbox_inches='tight', dpi=150)
    plt.close()
    plt.close()

    X_train_pca = compute_pca_features(pca_per_class, data_train, nb_components=nb_pcs_per_class)
    X_train_hog_v = []
    X_train_hog_h = []
    for img_flat in data_train:
        img_3d = img_flat.reshape((TARGET_SIZE[0], TARGET_SIZE[1], 3))
        img_hsv = color.rgb2hsv(img_3d) # CORRECTION : On convertit ici localement
        canal_v = img_hsv[:, :, 2]    # Extraction du canal V (Luminosité)
        canal_h= img_hsv[:,:,0]# Extraction du canal h (teinte)
       
        X_train_hog_v.append(compute_hog(canal_v, 4, 4, 8))
        X_train_hog_h.append(compute_hog(canal_h,4,4,8))
       
       
    X_train_hog_v = np.array(X_train_hog_v)
    X_train_hog_h=np.array(X_train_hog_h)
    X_train_combined = np.hstack((X_train_pca, X_train_hog_v,X_train_hog_h))
   
    # --- POUR LE TEST ---
    X_test_pca = compute_pca_features(pca_per_class, data_test, nb_components=nb_pcs_per_class)
    X_test_hog_v = []
    X_test_hog_h = []
    for img_flat in data_test:
        img_3d = img_flat.reshape((TARGET_SIZE[0], TARGET_SIZE[1], 3))
        img_hsv = color.rgb2hsv(img_3d)
        canal_v = img_hsv[:, :, 2]    # Extraction du canal V (Luminosité)
        canal_h = img_hsv[:, :, 0]    # Extraction du canal H (Teinte)
       
        X_test_hog_v.append(compute_hog(canal_v, 4, 4, 8))
        X_test_hog_h.append(compute_hog(canal_h, 4, 4, 8))
       
    # En dehors de la boucle for :
    X_test_hog_v = np.array(X_test_hog_v)
    X_test_hog_h = np.array(X_test_hog_h)
    X_test_combined = np.hstack((X_test_pca, X_test_hog_v, X_test_hog_h))
       
    print("\n Step 6: Standardizing Features & Optimizing Classifiers")
   
    # =========================================================================
    # TASK 4: ANALYSIS
    # =========================================================================
    # TODO: First, run this pipeline exactly as written to see the baseline performance.
    # TODO: Second, comment out the three lines immediately below this comment to
    # bypass the StandardScaler completely.
    # TODO: Third, pass the raw X_train_combined and X_test_combined matrices
    # directly into the knn.fit() and knn.predict() functions instead of the scaled versions.
    # Run the script again and observe what happens to the Testing Error Rate!
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)
 
    knn = KNeighborsClassifier(n_neighbors=5)
   
    # Update these variables if you disable the scaler above!
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
    # 1. Prédire les classes sur le jeu de test
    # (Pense à remplacer X_test_scaled par X_test_combined si tu désactives le scaler pour la Task 4)
    y_pred = knn.predict(X_test_scaled)
   
    # 2. Calculer la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
   
    # 3. Récupérer le nom des races de chien dans l'ordre des labels
    target_names = [label_names[i] for i in sorted(label_names.keys())]
   
    # 4. Affichage et sauvegarde de la matrice
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
   
    # On plot avec un dégradé de bleu et une rotation des labels pour la lisibilité
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=15)
    ax.set_title("Matrice de Confusion par Race de Chien", fontweight='bold', pad=15)
   
    # Sauvegarde du fichier
    plt.show()
    plt.savefig(os.path.join(FIGS, 'figures/matrice_confusion.png'), bbox_inches='tight', dpi=150)
   
    print("   - Confusion matrix successfully saved as 'figures/matrice_confusion.png'")