import numpy as np
from scipy.fft import dctn, idctn

# ============================================================
# MATRICE DE QUANTIFICATION STANDARD (inspirée du JPEG)
# Chaque valeur contrôle combien on compresse une fréquence
# - Valeurs faibles (coin haut-gauche) = fréquences basses = importantes (peu compressées)
# - Valeurs élevées (coin bas-droit) = hautes fréquences = détails fins (très compressées)
# ============================================================
QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)


def get_quant_matrix(quality=50):
    """
    Ajuste la matrice de quantification selon un facteur de qualité.
    
    quality=100 → très peu de compression, image quasi-originale
    quality=50  → compression modérée (valeur par défaut)
    quality=1   → compression maximale, image très dégradée
    
    C'est ce paramètre que tu vas faire varier dans ton rapport !
    """

    # Sécurité : on limite quality entre 1 et 100
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100

    # Calcul du facteur d'échelle selon la formule standard JPEG
    # quality < 50 : on augmente beaucoup la matrice (plus de compression)
    # quality > 50 : on réduit la matrice (moins de compression)
    if quality < 50:
        scale = 5000 / quality       # ex: quality=25 → scale=200
    else:
        scale = 200 - 2 * quality    # ex: quality=75 → scale=50

    # On multiplie chaque valeur de la matrice par le facteur d'échelle
    # np.floor arrondit vers le bas, +50 sert à arrondir correctement
    qm = np.floor((QUANT_MATRIX * scale + 50) / 100)

    # On s'assure que aucune valeur n'est 0 (division par 0) ni > 255
    qm = np.clip(qm, 1, 255)

    return qm.astype(np.float32)


def split_into_blocks(channel, block_size=8):
    """
    Découpe un canal 2D (ex: Y de taille 240x320) en blocs de 8x8.
    
    Pourquoi 8x8 ? C'est la taille standard utilisée par JPEG et MPEG.
    La DCT est plus efficace sur de petits blocs uniformes.
    
    Exemple : canal 240x320 → 30x40 blocs de 8x8 = 1200 blocs
    """

    h, w = channel.shape  # dimensions originales du canal

    # Si la taille n'est pas un multiple de 8, on ajoute du padding
    # ex: hauteur 235 → pad_h = 5 pour arriver à 240
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    # mode='edge' répète les pixels du bord (meilleur que mettre des zéros)
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')

    ph, pw = padded.shape  # dimensions après padding

    # Reshape intelligent pour créer les blocs sans boucle
    # padded (240, 320) → (30, 8, 40, 8)
    blocks = padded.reshape(ph // block_size, block_size,
                            pw // block_size, block_size)

    # Réorganise les axes pour avoir : (nb_blocs_h, nb_blocs_w, 8, 8)
    # ex: (30, 40, 8, 8) → 1200 blocs de 8x8
    blocks = blocks.transpose(0, 2, 1, 3)

    # On retourne aussi la taille originale pour supprimer le padding plus tard
    return blocks, (h, w)


def merge_blocks(blocks, original_shape, block_size=8):
    """
    Reconstruit le canal original à partir des blocs 8x8.
    C'est l'opération inverse de split_into_blocks().
    
    On recolle tous les blocs ensemble, puis on supprime le padding ajouté.
    """

    num_bh, num_bw, _, _ = blocks.shape  # ex: (30, 40, 8, 8)

    # Inverse du transpose fait dans split_into_blocks
    blocks = blocks.transpose(0, 2, 1, 3)  # → (30, 8, 40, 8)

    # Reshape pour recoller tous les blocs en une seule image
    channel = blocks.reshape(num_bh * block_size, num_bw * block_size)

    # Supprime le padding en recadrant à la taille originale
    h, w = original_shape
    return channel[:h, :w]


def dct_block(block):
    """
    Applique la DCT 2D sur un bloc 8x8.
    
    La DCT (Discrete Cosine Transform) convertit les pixels en fréquences :
    - Coefficient [0,0] (DC) = moyenne du bloc = information la plus importante
    - Autres coefficients (AC) = détails, textures, contours
    
    On soustrait 128 d'abord pour centrer les valeurs autour de 0
    (les pixels vont de 0 à 255, on les ramène à -128..+127)
    norm='ortho' normalise les coefficients pour qu'ils soient équilibrés
    """
    return dctn(block.astype(np.float32) - 128, norm='ortho')


def idct_block(block):
    """
    Applique la DCT inverse (IDCT) pour retrouver les pixels.
    
    C'est l'exact opposé de dct_block().
    On rajoute 128 à la fin pour revenir dans l'espace 0..255.
    """
    return idctn(block, norm='ortho') + 128


def quantise_block(dct_block, quant_matrix):
    """
    Quantifie un bloc DCT : divise chaque coefficient par la matrice
    et arrondit au nombre entier le plus proche.
    
    C'est l'étape AVEC PERTE — les petites valeurs deviennent 0.
    Plus la matrice est grande, plus de coefficients deviennent 0,
    donc plus on compresse (mais plus on perd de qualité).
    
    Exemple :
      coefficient DCT = 23.7, valeur matrice = 16
      → 23.7 / 16 = 1.48 → arrondi à 1  (on a perdu 0.48*16 = 7.68)
    
    int16 : entier signé sur 16 bits, suffit pour stocker les valeurs quantifiées
    """
    return np.round(dct_block / quant_matrix).astype(np.int16)


def dequantise_block(quant_block, quant_matrix):
    """
    Dequantifie un bloc : multiplie par la matrice pour retrouver
    une approximation des coefficients DCT originaux.
    
    On ne retrouve PAS exactement les valeurs originales à cause de l'arrondi,
    c'est pourquoi la compression est avec perte.
    
    Exemple (suite) :
      valeur quantifiée = 1, valeur matrice = 16
      → 1 * 16 = 16  (au lieu de 23.7 original → erreur de 7.7)
    """
    return (quant_block * quant_matrix).astype(np.float32)


def encode_channel(channel, quant_matrix):
    """
    Encode un canal complet (ex: Y, Cb ou Cr) en appliquant
    DCT + quantification sur chaque bloc 8x8.
    
    Retourne :
    - encoded : tableau de blocs quantifiés (int16)
    - original_shape : pour pouvoir supprimer le padding au décodage
    """

    # Étape 1 : découper en blocs 8x8
    blocks, original_shape = split_into_blocks(channel)
    num_bh, num_bw, _, _ = blocks.shape  # nombre de blocs en hauteur et largeur

    # Tableau vide pour stocker les blocs encodés
    encoded = np.zeros_like(blocks, dtype=np.int16)

    # Étape 2 : pour chaque bloc, appliquer DCT puis quantification
    for i in range(num_bh):       # parcourt les lignes de blocs
        for j in range(num_bw):   # parcourt les colonnes de blocs
            block = blocks[i, j]                        # bloc 8x8 de pixels
            dct = dct_block(block)                      # → coefficients fréquentiels
            encoded[i, j] = quantise_block(dct, quant_matrix)  # → entiers compressés

    return encoded, original_shape


def decode_channel(encoded_blocks, original_shape, quant_matrix):
    """
    Décode un canal complet en inversant la quantification et la DCT.
    
    Pour chaque bloc :
    1. Dequantification : retrouve approximation des coefficients DCT
    2. IDCT : retrouve les valeurs de pixels
    3. Clip : s'assure que les valeurs restent dans [0, 255]
    4. Merge : recolle tous les blocs
    """

    num_bh, num_bw, _, _ = encoded_blocks.shape
    decoded_blocks = np.zeros_like(encoded_blocks, dtype=np.float32)

    for i in range(num_bh):
        for j in range(num_bw):
            # Étape 1 : dequantification
            dequant = dequantise_block(encoded_blocks[i, j], quant_matrix)
            # Étape 2 : IDCT pour retrouver les pixels
            decoded_blocks[i, j] = idct_block(dequant)

    # Clip : les erreurs d'arrondi peuvent donner des valeurs < 0 ou > 255
    decoded_blocks = np.clip(decoded_blocks, 0, 255)

    # Recolle les blocs et supprime le padding
    return merge_blocks(decoded_blocks, original_shape)


def encode_iframe(Y, Cb, Cr, quality=50):
    """
    Encode une I-frame complète (les 3 canaux Y, Cb, Cr).
    
    Une I-frame est une frame codée de façon indépendante,
    sans référence aux autres frames (comme une image JPEG).
    
    Retourne :
    - les 3 canaux encodés
    - leurs formes originales (pour le décodage)
    - la matrice de quantification utilisée (pour le décodage)
    """

    # Calcule la matrice de quantification selon la qualité choisie
    qm = get_quant_matrix(quality)

    # Encode chaque canal séparément
    Y_enc,  Y_shape  = encode_channel(Y,  qm)   # canal luminance (pleine résolution)
    Cb_enc, Cb_shape = encode_channel(Cb, qm)   # canal bleu (demi-résolution)
    Cr_enc, Cr_shape = encode_channel(Cr, qm)   # canal rouge (demi-résolution)

    return (Y_enc, Cb_enc, Cr_enc), (Y_shape, Cb_shape, Cr_shape), qm


def decode_iframe(encoded_channels, shapes, quant_matrix):
    """
    Décode une I-frame complète (les 3 canaux).
    
    Inverse exactement encode_iframe() :
    pour chaque canal → dequantification → IDCT → reconstruction
    """

    Y_enc, Cb_enc, Cr_enc = encoded_channels
    Y_shape, Cb_shape, Cr_shape = shapes

    # Décode chaque canal indépendamment
    Y_dec  = decode_channel(Y_enc,  Y_shape,  quant_matrix)
    Cb_dec = decode_channel(Cb_enc, Cb_shape, quant_matrix)
    Cr_dec = decode_channel(Cr_enc, Cr_shape, quant_matrix)

    # Convertit en uint8 (entiers 0-255) pour les pixels finaux
    return Y_dec.astype(np.uint8), Cb_dec.astype(np.uint8), Cr_dec.astype(np.uint8)


# ============================================================
# EXÉCUTION PRINCIPALE
# ============================================================

QUALITY = 50  # Facteur de qualité : 1 (max compression) → 100 (meilleure qualité)
              # Fais varier cette valeur pour ton rapport !

encoded_iframes = []  # liste qui stocke toutes les I-frames encodées

# Encode chaque frame de la liste preprocessed (venant de la Partie 1)
for idx, (Y, Cb, Cr) in enumerate(preprocessed):
    encoded, shapes, qm = encode_iframe(Y, Cb, Cr, quality=QUALITY)
    encoded_iframes.append({
        "encoded": encoded,   # les 3 canaux encodés (coefficients DCT quantifiés)
        "shapes":  shapes,    # tailles originales (pour supprimer le padding)
        "qm":      qm         # matrice de quantification (nécessaire au décodage)
    })

print(f"Encodé {len(encoded_iframes)} I-frames")

# --- Vérification : décode la première frame et affiche les dimensions ---
first = encoded_iframes[0]
Y_dec, Cb_dec, Cr_dec = decode_iframe(first["encoded"], first["shapes"], first["qm"])

print(f"Y décodé  : {Y_dec.shape}")    # doit être la taille originale ex: (240, 320)
print(f"Cb décodé : {Cb_dec.shape}")   # demi-résolution ex: (120, 160)
print(f"Cr décodé : {Cr_dec.shape}")   # demi-résolution ex: (120, 160)