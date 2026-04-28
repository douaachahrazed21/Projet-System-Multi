import numpy as np
from scipy.fft import dctn, idctn

# ============================================================
# PARAMÈTRES PRINCIPAUX
# ============================================================

GOP_SIZE = 12       # Taille du GOP
SEARCH_WINDOW = 8   # Fenêtre de recherche ±8 pixels
BLOCK_SIZE = 16     # Taille des macroblocs : 16×16 pixels
QUALITY = 50        # Facteur de qualité

# Structure du GOP avec B-frames :
# I B B P B B P B B P B B I B B P ...
# Chaque P-frame est précédée de 2 B-frames
B_FRAME_COUNT = 2   # Nombre de B-frames entre chaque I/P frame


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
    qm_8 = get_quant_matrix(quality)
    qm_16 = np.kron(qm_8, np.ones((2, 2), dtype=np.float32))
    return qm_16


# ============================================================
# FONCTIONS DCT / IDCT
# ============================================================

def split_into_blocks(channel, block_size=8):
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
    num_bh, num_bw, _, _ = blocks.shape
    blocks = blocks.transpose(0, 2, 1, 3)
    channel = blocks.reshape(num_bh * block_size, num_bw * block_size)
    h, w = original_shape
    return channel[:h, :w]


def encode_channel(channel, quant_matrix):
    blocks, original_shape = split_into_blocks(channel, block_size=8)
    num_bh, num_bw, _, _ = blocks.shape
    encoded = np.zeros_like(blocks, dtype=np.int16)
    for i in range(num_bh):
        for j in range(num_bw):
            block = blocks[i, j]
            dct = dctn(block.astype(np.float32) - 128, norm='ortho')
            encoded[i, j] = np.round(dct / quant_matrix).astype(np.int16)
    return encoded, original_shape


def decode_channel(encoded_blocks, original_shape, quant_matrix):
    num_bh, num_bw, _, _ = encoded_blocks.shape
    decoded_blocks = np.zeros_like(encoded_blocks, dtype=np.float32)
    for i in range(num_bh):
        for j in range(num_bw):
            dequant = (encoded_blocks[i, j] * quant_matrix).astype(np.float32)
            decoded_blocks[i, j] = idctn(dequant, norm='ortho') + 128
    decoded_blocks = np.clip(decoded_blocks, 0, 255)
    return merge_blocks(decoded_blocks, original_shape, block_size=8)


# ============================================================
# FONCTIONS I-FRAME
# ============================================================

def encode_iframe(Y, Cb, Cr, quality=50):
    qm = get_quant_matrix(quality)
    Y_enc,  Y_shape  = encode_channel(Y,  qm)
    Cb_enc, Cb_shape = encode_channel(Cb, qm)
    Cr_enc, Cr_shape = encode_channel(Cr, qm)
    return (Y_enc, Cb_enc, Cr_enc), (Y_shape, Cb_shape, Cr_shape), qm


def decode_iframe(encoded_channels, shapes, quant_matrix):
    Y_enc, Cb_enc, Cr_enc = encoded_channels
    Y_shape, Cb_shape, Cr_shape = shapes
    Y_dec  = decode_channel(Y_enc,  Y_shape,  quant_matrix)
    Cb_dec = decode_channel(Cb_enc, Cb_shape, quant_matrix)
    Cr_dec = decode_channel(Cr_enc, Cr_shape, quant_matrix)
    return Y_dec.astype(np.uint8), Cb_dec.astype(np.uint8), Cr_dec.astype(np.uint8)


# ============================================================
# FONCTIONS UTILITAIRES MACROBLOCS
# ============================================================

def extract_block(frame, y, x, size=16):
    h, w = frame.shape
    y_end = min(y + size, h)
    x_end = min(x + size, w)
    block = frame[y:y_end, x:x_end]
    if block.shape != (size, size):
        padded = np.zeros((size, size), dtype=np.float32)
        padded[:block.shape[0], :block.shape[1]] = block
        return padded
    return block.astype(np.float32)


def compute_sad(block1, block2):
    return np.sum(np.abs(block1.astype(np.float32) -
                         block2.astype(np.float32)))


def find_best_match(current_block, ref_frame, y, x, search_window):
    h, w = ref_frame.shape
    best_sad = float('inf')
    best_dy  = 0
    best_dx  = 0
    for dy in range(-search_window, search_window + 1):
        for dx in range(-search_window, search_window + 1):
            ref_y = y + dy
            ref_x = x + dx
            if ref_y < 0 or ref_x < 0:
                continue
            if ref_y + BLOCK_SIZE > h or ref_x + BLOCK_SIZE > w:
                continue
            candidate = extract_block(ref_frame, ref_y, ref_x, BLOCK_SIZE)
            sad = compute_sad(current_block, candidate)
            if sad < best_sad:
                best_sad = sad
                best_dy  = dy
                best_dx  = dx
    return best_dy, best_dx


# ============================================================
# ENCODAGE / DÉCODAGE DU RÉSIDU 16×16
# ============================================================

def encode_residual(residual, qm_16):
    dct_coeffs = dctn(residual.astype(np.float32), norm='ortho')
    quantised = np.round(dct_coeffs / qm_16).astype(np.int16)
    return quantised


def decode_residual(quantised, qm_16):
    dct_coeffs = (quantised * qm_16).astype(np.float32)
    residual = idctn(dct_coeffs, norm='ortho')
    return residual


# ============================================================
# ENCODAGE / DÉCODAGE P-FRAME (inchangé)
# ============================================================

def encode_pframe(current_Y, ref_Y, quality):
    """
    Encode une P-frame avec motion estimation depuis UNE référence
    (la frame précédente reconstruite).
    """
    qm_16 = get_quant_matrix_16(quality)
    h, w = current_Y.shape
    motion_vectors = []
    residuals_enc  = []

    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):
            current_block   = extract_block(current_Y, y, x, BLOCK_SIZE)
            dy, dx          = find_best_match(current_block, ref_Y, y, x, SEARCH_WINDOW)
            motion_vectors.append((dy, dx))
            predicted_block = extract_block(ref_Y, y + dy, x + dx, BLOCK_SIZE)
            residual        = current_block - predicted_block
            enc_residual    = encode_residual(residual, qm_16)
            residuals_enc.append(enc_residual)

    return motion_vectors, residuals_enc


def decode_pframe(motion_vectors, residuals_enc, ref_Y, frame_shape, quality):
    qm_16 = get_quant_matrix_16(quality)
    h, w = frame_shape
    reconstructed_Y = np.zeros((h, w), dtype=np.float32)
    idx = 0

    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):
            dy, dx              = motion_vectors[idx]
            predicted_block     = extract_block(ref_Y, y + dy, x + dx, BLOCK_SIZE)
            residual            = decode_residual(residuals_enc[idx], qm_16)
            reconstructed_block = predicted_block + residual
            y_end = min(y + BLOCK_SIZE, h)
            x_end = min(x + BLOCK_SIZE, w)
            bh = y_end - y
            bw = x_end - x
            reconstructed_Y[y:y_end, x:x_end] = reconstructed_block[:bh, :bw]
            idx += 1

    return np.clip(reconstructed_Y, 0, 255).astype(np.uint8)


# ============================================================
# ENCODAGE / DÉCODAGE B-FRAME (NOUVEAU !)
# ============================================================

def encode_bframe(current_Y, ref_Y_past, ref_Y_future, quality):
    """
    Encode une B-frame avec interpolation bidirectionnelle.

    Contrairement aux P-frames qui utilisent UNE référence (le passé),
    les B-frames utilisent DEUX références :
    - ref_Y_past   : frame précédente reconstruite (ex: I ou P avant)
    - ref_Y_future : frame suivante reconstruite  (ex: P ou I après)

    Pour chaque macroblock :
    1. Cherche le meilleur match dans ref_Y_past  → motion vector forward
    2. Cherche le meilleur match dans ref_Y_future → motion vector backward
    3. Prédiction = moyenne des deux blocs trouvés (interpolation)
    4. Résidu = bloc courant - prédiction interpolée
    5. Encode le résidu avec DCT 16×16

    Pourquoi deux références ?
    → La prédiction est meilleure car on peut interpoler le mouvement
    → Les résidus sont plus petits → meilleure compression
    """
    qm_16 = get_quant_matrix_16(quality)
    h, w = current_Y.shape

    # Stocke les motion vectors forward (vers passé) et backward (vers futur)
    mv_forward  = []   # liste de (dy, dx) vers ref_past
    mv_backward = []   # liste de (dy, dx) vers ref_future
    residuals_enc = []

    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):

            # Étape 1 : extrait le bloc courant
            current_block = extract_block(current_Y, y, x, BLOCK_SIZE)

            # Étape 2 : motion estimation FORWARD (vers la frame passée)
            dy_fwd, dx_fwd = find_best_match(
                current_block, ref_Y_past, y, x, SEARCH_WINDOW
            )
            mv_forward.append((dy_fwd, dx_fwd))

            # Étape 3 : motion estimation BACKWARD (vers la frame future)
            dy_bwd, dx_bwd = find_best_match(
                current_block, ref_Y_future, y, x, SEARCH_WINDOW
            )
            mv_backward.append((dy_bwd, dx_bwd))

            # Étape 4 : extrait les deux blocs prédits
            block_from_past   = extract_block(ref_Y_past,   y + dy_fwd,
                                              x + dx_fwd, BLOCK_SIZE)
            block_from_future = extract_block(ref_Y_future, y + dy_bwd,
                                              x + dx_bwd, BLOCK_SIZE)

            # Étape 5 : interpolation bidirectionnelle
            # Moyenne des deux prédictions → résidu encore plus petit !
            predicted_block = (block_from_past + block_from_future) / 2.0

            # Étape 6 : calcule et encode le résidu
            residual = current_block - predicted_block
            enc_residual = encode_residual(residual, qm_16)
            residuals_enc.append(enc_residual)

    return mv_forward, mv_backward, residuals_enc


def decode_bframe(mv_forward, mv_backward, residuals_enc,
                  ref_Y_past, ref_Y_future, frame_shape, quality):
    """
    Reconstruit une B-frame depuis les motion vectors et résidus.

    Pour chaque macroblock :
    1. Extrait bloc prédit depuis ref_Y_past  avec mv_forward
    2. Extrait bloc prédit depuis ref_Y_future avec mv_backward
    3. Interpolation : moyenne des deux
    4. Décode le résidu
    5. bloc reconstruit = interpolation + résidu décodé
    """
    qm_16 = get_quant_matrix_16(quality)
    h, w = frame_shape
    reconstructed_Y = np.zeros((h, w), dtype=np.float32)
    idx = 0

    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):

            # Étape 1 : récupère les motion vectors
            dy_fwd, dx_fwd = mv_forward[idx]
            dy_bwd, dx_bwd = mv_backward[idx]

            # Étape 2 : extrait les deux blocs prédits
            block_from_past   = extract_block(ref_Y_past,   y + dy_fwd,
                                              x + dx_fwd, BLOCK_SIZE)
            block_from_future = extract_block(ref_Y_future, y + dy_bwd,
                                              x + dx_bwd, BLOCK_SIZE)

            # Étape 3 : interpolation bidirectionnelle
            predicted_block = (block_from_past + block_from_future) / 2.0

            # Étape 4 : décode le résidu
            residual = decode_residual(residuals_enc[idx], qm_16)

            # Étape 5 : reconstruit = prédiction + résidu
            reconstructed_block = predicted_block + residual

            # Étape 6 : place le bloc dans l'image
            y_end = min(y + BLOCK_SIZE, h)
            x_end = min(x + BLOCK_SIZE, w)
            bh = y_end - y
            bw = x_end - x
            reconstructed_Y[y:y_end, x:x_end] = reconstructed_block[:bh, :bw]

            idx += 1

    return np.clip(reconstructed_Y, 0, 255).astype(np.uint8)


# ============================================================
# PIPELINE COMPLET : ENCODE TOUTES LES FRAMES (I + P + B)
# ============================================================

def encode_video(preprocessed, quality=QUALITY):
    """
    Encode toutes les frames avec la structure GOP suivante :

    Avec GOP_SIZE=12 et B_FRAME_COUNT=2 :
    Position : 0  1  2  3  4  5  6  7  8  9  10 11
    Type      : I  B  B  P  B  B  P  B  B  P  B  B

    IMPORTANT sur l'ordre d'encodage des B-frames :
    Les B-frames ont besoin de leur frame future comme référence.
    Donc on encode d'abord les I/P frames, puis les B-frames entre elles.

    On utilise un buffer pour stocker les frames en attente
    jusqu'à ce qu'on ait la référence future disponible.
    """
    qm = get_quant_matrix(quality)
    encoded_frames = []

    # Dictionnaire pour stocker les frames reconstruites
    # clé = index de frame, valeur = canal Y reconstruit
    reconstructed_refs = {}

    total = len(preprocessed)

    # ---- Étape 1 : identifier le type de chaque frame ----
    frame_types = []
    for idx in range(total):
        pos_in_gop = idx % GOP_SIZE
        if pos_in_gop == 0:
            # Première frame du GOP → toujours I-frame
            frame_types.append('I')
        elif (pos_in_gop % (B_FRAME_COUNT + 1)) == 0:
            # Toutes les (B_FRAME_COUNT+1) positions → P-frame
            frame_types.append('P')
        else:
            # Le reste → B-frame
            frame_types.append('B')

    print("Structure GOP détectée :")
    print(' '.join(frame_types[:GOP_SIZE]))

    # ---- Étape 2 : encoder dans l'ordre de décodage ----
    # On encode les I et P d'abord pour avoir les références,
    # puis on encode les B-frames entre elles

    # Buffer pour stocker les données encodées dans l'ordre original
    encoded_buffer = [None] * total

    # Passe 1 : encode toutes les I-frames et P-frames
    ref_Y = None
    for idx in range(total):
        Y, Cb, Cr = preprocessed[idx]

        if frame_types[idx] == 'I':
            print(f"Frame {idx} → I-frame")
            encoded, shapes, _ = encode_iframe(Y, Cb, Cr, quality)
            Y_rec, _, _ = decode_iframe(encoded, shapes, qm)
            ref_Y = Y_rec.astype(np.float32)
            reconstructed_refs[idx] = ref_Y

            encoded_buffer[idx] = {
                "type":    "I",
                "encoded": encoded,
                "shapes":  shapes,
                "qm":      qm
            }

        elif frame_types[idx] == 'P':
            print(f"Frame {idx} → P-frame")
            motion_vectors, residuals_enc = encode_pframe(
                Y.astype(np.float32), ref_Y, quality
            )
            Cb_enc, Cb_shape = encode_channel(Cb, qm)
            Cr_enc, Cr_shape = encode_channel(Cr, qm)

            Y_rec = decode_pframe(
                motion_vectors, residuals_enc, ref_Y, Y.shape, quality
            )
            ref_Y = Y_rec.astype(np.float32)
            reconstructed_refs[idx] = ref_Y

            encoded_buffer[idx] = {
                "type":           "P",
                "motion_vectors": motion_vectors,
                "residuals_enc":  residuals_enc,
                "Cb_enc":         Cb_enc,
                "Cr_enc":         Cr_enc,
                "Cb_shape":       Cb_shape,
                "Cr_shape":       Cr_shape,
                "frame_shape":    Y.shape,
                "qm":             qm
            }

    # Passe 2 : encode toutes les B-frames
    # Maintenant on a toutes les références I/P disponibles
    for idx in range(total):
        if frame_types[idx] != 'B':
            continue

        Y, Cb, Cr = preprocessed[idx]
        print(f"Frame {idx} → B-frame")

        # Trouve la référence PASSÉE (I ou P précédente)
        past_ref_idx = idx - 1
        while past_ref_idx >= 0 and frame_types[past_ref_idx] == 'B':
            past_ref_idx -= 1

        # Trouve la référence FUTURE (I ou P suivante)
        future_ref_idx = idx + 1
        while future_ref_idx < total and frame_types[future_ref_idx] == 'B':
            future_ref_idx += 1

        # Vérifie qu'on a bien les deux références
        if past_ref_idx < 0 or future_ref_idx >= total:
            # Si pas de référence future → encode comme P-frame
            print(f"  → pas de ref future, encodée comme P-frame")
            past_ref = reconstructed_refs.get(past_ref_idx,
                       reconstructed_refs[min(reconstructed_refs.keys())])
            motion_vectors, residuals_enc = encode_pframe(
                Y.astype(np.float32), past_ref, quality
            )
            Cb_enc, Cb_shape = encode_channel(Cb, qm)
            Cr_enc, Cr_shape = encode_channel(Cr, qm)

            encoded_buffer[idx] = {
                "type":           "P",
                "motion_vectors": motion_vectors,
                "residuals_enc":  residuals_enc,
                "Cb_enc":         Cb_enc,
                "Cr_enc":         Cr_enc,
                "Cb_shape":       Cb_shape,
                "Cr_shape":       Cr_shape,
                "frame_shape":    Y.shape,
                "qm":             qm
            }
            continue

        # Récupère les deux références reconstruites
        ref_Y_past   = reconstructed_refs[past_ref_idx]
        ref_Y_future = reconstructed_refs[future_ref_idx]

        # Encode la B-frame avec interpolation bidirectionnelle
        mv_forward, mv_backward, residuals_enc = encode_bframe(
            Y.astype(np.float32),
            ref_Y_past,
            ref_Y_future,
            quality
        )

        # Encode Cb et Cr comme I-frames
        Cb_enc, Cb_shape = encode_channel(Cb, qm)
        Cr_enc, Cr_shape = encode_channel(Cr, qm)

        encoded_buffer[idx] = {
            "type":           "B",
            "mv_forward":     mv_forward,     # motion vectors vers le passé
            "mv_backward":    mv_backward,    # motion vectors vers le futur
            "residuals_enc":  residuals_enc,
            "past_ref_idx":   past_ref_idx,   # index de la référence passée
            "future_ref_idx": future_ref_idx, # index de la référence future
            "Cb_enc":         Cb_enc,
            "Cr_enc":         Cr_enc,
            "Cb_shape":       Cb_shape,
            "Cr_shape":       Cr_shape,
            "frame_shape":    Y.shape,
            "qm":             qm
        }

    # Vérifie qu'aucune frame n'est None
    encoded_frames = [f for f in encoded_buffer if f is not None]

    return encoded_frames, reconstructed_refs


# ============================================================
# DÉCODAGE COMPLET (I + P + B)
# ============================================================

def decode_video(encoded_frames, reconstructed_refs):
    """
    Décode toutes les frames dans l'ordre original.

    Pour les B-frames, on utilise reconstructed_refs qui contient
    déjà les références I/P reconstruites pendant l'encodage.
    (En vrai décodeur, on les recalculerait à la volée)
    """
    decoded_frames = []
    ref_Y = None

    for idx, frame_data in enumerate(encoded_frames):

        if frame_data["type"] == "I":
            # ---- Décode I-frame ----
            Y_dec, Cb_dec, Cr_dec = decode_iframe(
                frame_data["encoded"],
                frame_data["shapes"],
                frame_data["qm"]
            )
            ref_Y = Y_dec.astype(np.float32)

        elif frame_data["type"] == "P":
            # ---- Décode P-frame ----
            qm = frame_data["qm"]
            Y_dec = decode_pframe(
                frame_data["motion_vectors"],
                frame_data["residuals_enc"],
                ref_Y,
                frame_data["frame_shape"],
                QUALITY
            )
            ref_Y = Y_dec.astype(np.float32)
            Cb_dec = decode_channel(frame_data["Cb_enc"], frame_data["Cb_shape"], qm)
            Cr_dec = decode_channel(frame_data["Cr_enc"], frame_data["Cr_shape"], qm)
            Y_dec  = Y_dec.astype(np.uint8)
            Cb_dec = Cb_dec.astype(np.uint8)
            Cr_dec = Cr_dec.astype(np.uint8)

        else:
            # ---- Décode B-frame ----
            qm = frame_data["qm"]

            # Récupère les deux références depuis reconstructed_refs
            ref_Y_past   = reconstructed_refs[frame_data["past_ref_idx"]]
            ref_Y_future = reconstructed_refs[frame_data["future_ref_idx"]]

            # Reconstruit Y avec interpolation bidirectionnelle
            Y_dec = decode_bframe(
                frame_data["mv_forward"],
                frame_data["mv_backward"],
                frame_data["residuals_enc"],
                ref_Y_past,
                ref_Y_future,
                frame_data["frame_shape"],
                QUALITY
            )

            # Décode Cb et Cr
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
encoded_frames, reconstructed_refs = encode_video(preprocessed, quality=QUALITY)

print(f"\nTotal frames encodées : {len(encoded_frames)}")
print(f"I-frames : {sum(1 for f in encoded_frames if f['type'] == 'I')}")
print(f"P-frames : {sum(1 for f in encoded_frames if f['type'] == 'P')}")
print(f"B-frames : {sum(1 for f in encoded_frames if f['type'] == 'B')}")

# Décode toutes les frames pour vérification
decoded_frames = decode_video(encoded_frames, reconstructed_refs)
print(f"\nTotal frames décodées : {len(decoded_frames)}")