import numpy as np
from scipy.fft import dctn, idctn

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

    # Sécurité : on limite quality entre 1 et 100
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100

    # Calcul du facteur d'échelle selon la formule standard JPEG
    if quality < 50:
        scale = 5000 / quality      
    else:
        scale = 200 - 2 * quality   

    # On multiplie chaque valeur de la matrice par le facteur d'échelle
    # np.floor arrondit vers le bas, +50 sert à arrondir correctement
    qm = np.floor((QUANT_MATRIX * scale + 50) / 100)

    # On s'assure que aucune valeur n'est 0  ni > 255
    qm = np.clip(qm, 1, 255)

    return qm.astype(np.float32)


def split_into_blocks(channel, block_size=8):

    h, w = channel.shape  # dimensions originales du canal

    # Si la taille n'est pas un multiple de 8, on ajoute du padding
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
    blocks = blocks.transpose(0, 2, 1, 3)

    # On retourne aussi la taille originale pour supprimer le padding plus tard
    return blocks, (h, w)


def merge_blocks(blocks, original_shape, block_size=8):

    num_bh, num_bw, _, _ = blocks.shape  

    # Inverse du transpose fait dans split_into_blocks
    blocks = blocks.transpose(0, 2, 1, 3)  

    # Reshape pour recoller tous les blocs en une seule image
    channel = blocks.reshape(num_bh * block_size, num_bw * block_size)

    # Supprime le padding en recadrant à la taille originale
    h, w = original_shape
    return channel[:h, :w]


def dct_block(block):

    return dctn(block.astype(np.float32) - 128, norm='ortho')


def idct_block(block):

    return idctn(block, norm='ortho') + 128


def quantise_block(dct_block, quant_matrix):

    return np.round(dct_block / quant_matrix).astype(np.int16)


def dequantise_block(quant_block, quant_matrix):

    return (quant_block * quant_matrix).astype(np.float32)


def encode_channel(channel, quant_matrix):

    # Étape 1 : découper en blocs 8x8
    blocks, original_shape = split_into_blocks(channel)
    num_bh, num_bw, _, _ = blocks.shape  

    # Tableau vide pour stocker les blocs encodés
    encoded = np.zeros_like(blocks, dtype=np.int16)

    # Étape 2 : pour chaque bloc, appliquer DCT puis quantification
    for i in range(num_bh):      
        for j in range(num_bw):   
            block = blocks[i, j]                        
            dct = dct_block(block)                      
            encoded[i, j] = quantise_block(dct, quant_matrix)  

    return encoded, original_shape


def decode_channel(encoded_blocks, original_shape, quant_matrix):

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

    # Calcule la matrice de quantification selon la qualité choisie
    qm = get_quant_matrix(quality)

    # Encode chaque canal séparément
    Y_enc,  Y_shape  = encode_channel(Y,  qm)   
    Cb_enc, Cb_shape = encode_channel(Cb, qm)  
    Cr_enc, Cr_shape = encode_channel(Cr, qm)   

    return (Y_enc, Cb_enc, Cr_enc), (Y_shape, Cb_shape, Cr_shape), qm


def decode_iframe(encoded_channels, shapes, quant_matrix):

    Y_enc, Cb_enc, Cr_enc = encoded_channels
    Y_shape, Cb_shape, Cr_shape = shapes

    # Décode chaque canal indépendamment
    Y_dec  = decode_channel(Y_enc,  Y_shape,  quant_matrix)
    Cb_dec = decode_channel(Cb_enc, Cb_shape, quant_matrix)
    Cr_dec = decode_channel(Cr_enc, Cr_shape, quant_matrix)

    # Convertit en uint8 (entiers 0-255) pour les pixels finaux
    return Y_dec.astype(np.uint8), Cb_dec.astype(np.uint8), Cr_dec.astype(np.uint8)



QUALITY = 50  # Facteur de qualité : 1 (max compression) → 100 (meilleure qualité)
              # Fais varier cette valeur pour ton rapport !

encoded_iframes = []  # liste qui stocke toutes les I-frames encodées

# Encode chaque frame de la liste preprocessed (venant de la Partie 1)
for idx, (Y, Cb, Cr) in enumerate(preprocessed):
    encoded, shapes, qm = encode_iframe(Y, Cb, Cr, quality=QUALITY)
    encoded_iframes.append({
        "encoded": encoded,   
        "shapes":  shapes,   
        "qm":      qm      
    })

print(f"Encodé {len(encoded_iframes)} I-frames")

# --- Vérification : décode la première frame et affiche les dimensions ---
first = encoded_iframes[0]
Y_dec, Cb_dec, Cr_dec = decode_iframe(first["encoded"], first["shapes"], first["qm"])

print(f"Y décodé  : {Y_dec.shape}")   
print(f"Cb décodé : {Cb_dec.shape}")  
print(f"Cr décodé : {Cr_dec.shape}")   