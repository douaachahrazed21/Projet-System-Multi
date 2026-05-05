import numpy as np
from scipy.fft import dctn, idctn

# ============================================================
# PARAMÈTRES PRINCIPAUX
# ============================================================

GOP_SIZE = 5        # Toutes les 5 frames → I-frame, le reste → P-frames
SEARCH_WINDOW = 8   # Cherche le meilleur bloc dans ±8 pixels
BLOCK_SIZE = 16     # Taille des macroblocs : 16×16 pixels
QUALITY = 50        # Facteur de qualité pour la quantification


# ============================================================
# MATRICES DE QUANTIFICATION
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
    Retourne la matrice de quantification 8×8 selon le facteur de qualité.
    quality=100 → peu de compression
    quality=1   → compression maximale
    """
    if quality <= 0: quality = 1
    if quality > 100: quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    qm = np.floor((QUANT_MATRIX * scale + 50) / 100)
    qm = np.clip(qm, 1, 255)
    return qm.astype(np.float32)


def get_quant_matrix_16(quality=50):
    """
    Retourne une matrice de quantification 16×16 pour les résidus des P-frames.
    On agrandit la matrice 8×8 en répétant chaque valeur 2 fois
    en hauteur et en largeur avec np.kron.

    Exemple :
    [[16, 11],      [[16, 16, 11, 11],
     [12, 12]]  →    [16, 16, 11, 11],
                     [12, 12, 12, 12],
                     [12, 12, 12, 12]]
    """
    qm_8 = get_quant_matrix(quality)
    qm_16 = np.kron(qm_8, np.ones((2, 2), dtype=np.float32))
    return qm_16


# ============================================================
# FONCTIONS DCT / IDCT (réutilisées de la Part 2)
# ============================================================

def dct_block(block):
    """
    Applique la DCT 2D sur un bloc.
    Soustrait 128 pour centrer les valeurs autour de 0.
    """
    return dctn(block.astype(np.float32) - 128, norm='ortho')


def idct_block(block):
    """
    Applique la DCT inverse (IDCT).
    Rajoute 128 pour revenir dans l'espace 0..255.
    """
    return idctn(block, norm='ortho') + 128


def split_into_blocks(channel, block_size=8):
    """Découpe un canal 2D en blocs de block_size × block_size."""
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
    ph, pw = padded.shape
    blocks = padded.reshape(ph // block_size, block_size,
                            pw // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3)
    return blocks, (h, w)


def merge_blocks(blocks, original_shape, block_size=8):
    """Reconstruit un canal à partir de blocs."""
    num_bh, num_bw, _, _ = blocks.shape
    blocks = blocks.transpose(0, 2, 1, 3)
    channel = blocks.reshape(num_bh * block_size, num_bw * block_size)
    h, w = original_shape
    return channel[:h, :w]


def encode_channel(channel, quant_matrix):
    """
    Encode un canal complet avec DCT + quantification (8×8 blocs).
    Utilisé pour les I-frames et pour Cb/Cr des P-frames.
    """
    blocks, original_shape = split_into_blocks(channel, block_size=8)
    num_bh, num_bw, _, _ = blocks.shape
    encoded = np.zeros_like(blocks, dtype=np.int16)

    for i in range(num_bh):
        for j in range(num_bw):
            block = blocks[i, j]
            # Soustrait 128, applique DCT, quantifie
            dct = dctn(block.astype(np.float32) - 128, norm='ortho')
            encoded[i, j] = np.round(dct / quant_matrix).astype(np.int16)

    return encoded, original_shape


def decode_channel(encoded_blocks, original_shape, quant_matrix):
    """
    Décode un canal complet : dequantification + IDCT.
    """
    num_bh, num_bw, _, _ = encoded_blocks.shape
    decoded_blocks = np.zeros_like(encoded_blocks, dtype=np.float32)

    for i in range(num_bh):
        for j in range(num_bw):
            # Dequantifie puis applique IDCT
            dequant = (encoded_blocks[i, j] * quant_matrix).astype(np.float32)
            decoded_blocks[i, j] = idctn(dequant, norm='ortho') + 128

    decoded_blocks = np.clip(decoded_blocks, 0, 255)
    return merge_blocks(decoded_blocks, original_shape, block_size=8)


# ============================================================
# FONCTIONS I-FRAME (réutilisées de la Part 2)
# ============================================================

def encode_iframe(Y, Cb, Cr, quality=50):
    """Encode une I-frame complète (Y, Cb, Cr) avec DCT 8×8."""
    qm = get_quant_matrix(quality)
    Y_enc,  Y_shape  = encode_channel(Y,  qm)
    Cb_enc, Cb_shape = encode_channel(Cb, qm)
    Cr_enc, Cr_shape = encode_channel(Cr, qm)
    return (Y_enc, Cb_enc, Cr_enc), (Y_shape, Cb_shape, Cr_shape), qm


def decode_iframe(encoded_channels, shapes, quant_matrix):
    """Décode une I-frame complète."""
    Y_enc, Cb_enc, Cr_enc = encoded_channels
    Y_shape, Cb_shape, Cr_shape = shapes
    Y_dec  = decode_channel(Y_enc,  Y_shape,  quant_matrix)
    Cb_dec = decode_channel(Cb_enc, Cb_shape, quant_matrix)
    Cr_dec = decode_channel(Cr_enc, Cr_shape, quant_matrix)
    return Y_dec.astype(np.uint8), Cb_dec.astype(np.uint8), Cr_dec.astype(np.uint8)


# ============================================================
# FONCTIONS UTILITAIRES POUR LES MACROBLOCS 16×16
# ============================================================

def extract_block(frame, y, x, size=16):
    """
    Extrait un bloc de taille size×size depuis une frame.
    Si le bloc dépasse les bords, on complète avec des zéros.
    """
    h, w = frame.shape
    y_end = min(y + size, h)
    x_end = min(x + size, w)
    block = frame[y:y_end, x:x_end]

    # Complète avec des zéros si le bloc est incomplet (bords de l'image)
    if block.shape != (size, size):
        padded = np.zeros((size, size), dtype=np.float32)
        padded[:block.shape[0], :block.shape[1]] = block
        return padded

    return block.astype(np.float32)


def compute_sad(block1, block2):
    """
    Calcule la SAD (Sum of Absolute Differences) entre deux blocs.
    SAD = 0        → blocs identiques
    SAD très grand → blocs très différents
    On cherche toujours le bloc avec la SAD MINIMALE.
    """
    return np.sum(np.abs(block1.astype(np.float32) -
                         block2.astype(np.float32)))


# ============================================================
# ESTIMATION DE MOUVEMENT
# ============================================================

def find_best_match(current_block, ref_frame, y, x, search_window):
    """
    Cherche le bloc le plus similaire dans la frame de référence.

    Parcourt toutes les positions dans la fenêtre ±search_window
    et retourne le motion vector (dy, dx) du meilleur match.
    """
    h, w = ref_frame.shape
    best_sad = float('inf')
    best_dy  = 0
    best_dx  = 0

    for dy in range(-search_window, search_window + 1):
        for dx in range(-search_window, search_window + 1):

            ref_y = y + dy
            ref_x = x + dx

            # Ignore les positions hors de l'image
            if ref_y < 0 or ref_x < 0:
                continue
            if ref_y + BLOCK_SIZE > h or ref_x + BLOCK_SIZE > w:
                continue

            # Extrait le bloc candidat et calcule la SAD
            candidate = extract_block(ref_frame, ref_y, ref_x, BLOCK_SIZE)
            sad = compute_sad(current_block, candidate)

            # Garde le meilleur candidat
            if sad < best_sad:
                best_sad = sad
                best_dy  = dy
                best_dx  = dx

    return best_dy, best_dx


# ============================================================
# ENCODAGE / DÉCODAGE DU RÉSIDU 16×16
# ============================================================

def encode_residual(residual, qm_16):
    """
    Encode le résidu 16×16 avec DCT + quantification.
    Pas de -128 car le résidu est déjà centré autour de 0.
    qm_16 doit être une matrice 16×16 (pas 8×8 !)
    """
    # DCT sur le résidu 16×16
    dct_coeffs = dctn(residual.astype(np.float32), norm='ortho')

    # Quantification avec la matrice 16×16
    quantised = np.round(dct_coeffs / qm_16).astype(np.int16)

    return quantised


def decode_residual(quantised, qm_16):
    """
    Décode le résidu : dequantification + IDCT.
    qm_16 doit être une matrice 16×16 (pas 8×8 !)
    """
    # Dequantification
    dct_coeffs = (quantised * qm_16).astype(np.float32)

    # IDCT pour retrouver le résidu spatial
    residual = idctn(dct_coeffs, norm='ortho')

    return residual


# ============================================================
# ENCODAGE D'UNE P-FRAME
# ============================================================

def encode_pframe(current_Y, ref_Y, quality):
    """
    Encode une P-frame (canal Y uniquement pour motion estimation).

    Pour chaque macroblock 16×16 :
    1. Trouve le meilleur match dans ref_Y → motion vector (dy, dx)
    2. Calcule le résidu = bloc courant - bloc prédit
    3. Encode le résidu avec DCT 16×16 + quantification

    Retourne :
    - motion_vectors : liste de (dy, dx) pour chaque macroblock
    - residuals_enc  : liste des résidus encodés
    """
    # Matrice 16×16 pour les résidus
    qm_16 = get_quant_matrix_16(quality)

    h, w = current_Y.shape
    motion_vectors = []
    residuals_enc  = []

    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):

            # Étape 1 : extrait le bloc courant 16×16
            current_block = extract_block(current_Y, y, x, BLOCK_SIZE)

            # Étape 2 : motion estimation → trouve (dy, dx)
            dy, dx = find_best_match(current_block, ref_Y, y, x, SEARCH_WINDOW)
            motion_vectors.append((dy, dx))

            # Étape 3 : extrait le bloc prédit depuis la référence
            predicted_block = extract_block(ref_Y, y + dy, x + dx, BLOCK_SIZE)

            # Étape 4 : calcule le résidu
            # Si prédiction parfaite → résidu = 0 → très compressible !
            residual = current_block - predicted_block

            # Étape 5 : encode le résidu avec DCT 16×16
            enc_residual = encode_residual(residual, qm_16)
            residuals_enc.append(enc_residual)

    return motion_vectors, residuals_enc


# ============================================================
# DÉCODAGE D'UNE P-FRAME
# ============================================================

def decode_pframe(motion_vectors, residuals_enc, ref_Y, frame_shape, quality):
    """
    Reconstruit une P-frame depuis les motion vectors et résidus.

    Pour chaque macroblock :
    1. Utilise le motion vector pour extraire le bloc prédit
    2. Décode le résidu
    3. bloc reconstruit = bloc prédit + résidu décodé
    """
    # Matrice 16×16 pour les résidus
    qm_16 = get_quant_matrix_16(quality)

    h, w = frame_shape
    reconstructed_Y = np.zeros((h, w), dtype=np.float32)
    idx = 0

    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):

            # Étape 1 : récupère le motion vector
            dy, dx = motion_vectors[idx]

            # Étape 2 : extrait le bloc prédit depuis la référence
            predicted_block = extract_block(ref_Y, y + dy, x + dx, BLOCK_SIZE)

            # Étape 3 : décode le résidu
            residual = decode_residual(residuals_enc[idx], qm_16)

            # Étape 4 : reconstruit = prédiction + résidu
            reconstructed_block = predicted_block + residual

            # Étape 5 : place le bloc dans l'image reconstruite
            y_end = min(y + BLOCK_SIZE, h)
            x_end = min(x + BLOCK_SIZE, w)
            bh = y_end - y
            bw = x_end - x
            reconstructed_Y[y:y_end, x:x_end] = reconstructed_block[:bh, :bw]

            idx += 1

    return np.clip(reconstructed_Y, 0, 255).astype(np.uint8)


# ============================================================
# PIPELINE COMPLET : ENCODE TOUTES LES FRAMES
# ============================================================

def encode_video(preprocessed, quality=QUALITY):
    """
    Encode toutes les frames en mélangeant I-frames et P-frames.

    GOP_SIZE=5 → structure :
    Frame 0 → I
    Frame 1 → P (référence : frame 0 reconstruite)
    Frame 2 → P (référence : frame 1 reconstruite)
    Frame 3 → P (référence : frame 2 reconstruite)
    Frame 4 → P (référence : frame 3 reconstruite)
    Frame 5 → I
    ...

    IMPORTANT : les P-frames utilisent toujours la frame
    RECONSTRUITE comme référence (pas l'originale) pour que
    l'encodeur et le décodeur restent synchronisés.
    """
    qm = get_quant_matrix(quality)
    encoded_frames = []
    ref_Y = None  # canal Y de la dernière frame reconstruite

    for idx, (Y, Cb, Cr) in enumerate(preprocessed):

        is_iframe = (idx % GOP_SIZE == 0)

        if is_iframe:
            # ---- I-FRAME ----
            print(f"Frame {idx} → I-frame")

            # Encode les 3 canaux (DCT 8×8)
            encoded, shapes, _ = encode_iframe(Y, Cb, Cr, quality)

            # Décode immédiatement pour avoir la référence reconstruite
            Y_rec, Cb_rec, Cr_rec = decode_iframe(encoded, shapes, qm)
            ref_Y = Y_rec.astype(np.float32)

            encoded_frames.append({
                "type":    "I",
                "encoded": encoded,   # (Y_enc, Cb_enc, Cr_enc)
                "shapes":  shapes,    # (Y_shape, Cb_shape, Cr_shape)
                "qm":      qm
            })

        else:
            # ---- P-FRAME ----
            print(f"Frame {idx} → P-frame")

            # Encode Y avec motion estimation + résidus 16×16
            motion_vectors, residuals_enc = encode_pframe(
                Y.astype(np.float32),
                ref_Y,
                quality
            )

            # Encode Cb et Cr comme des I-frames (DCT 8×8)
            # La compensation de mouvement sur Cb/Cr est optionnelle
            Cb_enc, Cb_shape = encode_channel(Cb, qm)
            Cr_enc, Cr_shape = encode_channel(Cr, qm)

            # Décode Y immédiatement pour mettre à jour la référence
            Y_rec = decode_pframe(
                motion_vectors,
                residuals_enc,
                ref_Y,
                Y.shape,
                quality
            )
            ref_Y = Y_rec.astype(np.float32)

            encoded_frames.append({
                "type":           "P",
                "motion_vectors": motion_vectors,  # liste de (dy, dx)
                "residuals_enc":  residuals_enc,   # résidus DCT 16×16 quantifiés
                "Cb_enc":         Cb_enc,
                "Cr_enc":         Cr_enc,
                "Cb_shape":       Cb_shape,
                "Cr_shape":       Cr_shape,
                "frame_shape":    Y.shape,
                "qm":             qm
            })

    return encoded_frames


# ============================================================
# DÉCODAGE COMPLET DE LA VIDÉO
# ============================================================

def decode_video(encoded_frames):
    """
    Décode toutes les frames (I et P) pour reconstruire la vidéo.
    Retourne une liste de tuples (Y, Cb, Cr) reconstruits.
    """
    decoded_frames = []
    ref_Y = None

    for idx, frame_data in enumerate(encoded_frames):

        if frame_data["type"] == "I":
            # ---- Décode une I-frame ----
            Y_dec, Cb_dec, Cr_dec = decode_iframe(
                frame_data["encoded"],
                frame_data["shapes"],
                frame_data["qm"]
            )
            ref_Y = Y_dec.astype(np.float32)

        else:
            # ---- Décode une P-frame ----
            qm = frame_data["qm"]

            # Reconstruit Y avec motion compensation + résidu
            Y_dec = decode_pframe(
                frame_data["motion_vectors"],
                frame_data["residuals_enc"],
                ref_Y,
                frame_data["frame_shape"],
                QUALITY
            )
            ref_Y = Y_dec.astype(np.float32)

            # Décode Cb et Cr (encodés comme I-frames)
            Cb_dec = decode_channel(frame_data["Cb_enc"], frame_data["Cb_shape"], qm)
            Cr_dec = decode_channel(frame_data["Cr_enc"], frame_data["Cr_shape"], qm)

            Y_dec  = Y_dec.astype(np.uint8)
            Cb_dec = Cb_dec.astype(np.uint8)
            Cr_dec = Cr_dec.astype(np.uint8)

        decoded_frames.append((Y_dec, Cb_dec, Cr_dec))
        print(f"Frame {idx} ({frame_data['type']}) décodée ✓")

    return decoded_frames


# ============================================================
# EXÉCUTION
# ============================================================

# Encode toutes les frames
encoded_frames = encode_video(preprocessed, quality=QUALITY)

print(f"\nTotal frames encodées : {len(encoded_frames)}")
print(f"I-frames : {sum(1 for f in encoded_frames if f['type'] == 'I')}")
print(f"P-frames : {sum(1 for f in encoded_frames if f['type'] == 'P')}")

# Décode toutes les frames pour vérification
decoded_frames = decode_video(encoded_frames)
print(f"\nTotal frames décodées : {len(decoded_frames)}")