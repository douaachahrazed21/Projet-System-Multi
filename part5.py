import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fft import dctn, idctn

########################################## PART 5 : EVALUATION & VISUALISATION ##############################################
# Requires: preprocessed, frames_bgr (Part 1), encoded_frames, decoded_frames (Part 3),
#           original_size, compressed_size (Part 4)


# ---- 5A : Quality Metrics ----

# preprocessed, decoded_frames, encoded_frames, tailles → affichage métriques + dict résultats
def evaluate_pipeline(preprocessed, decoded_frames, encoded_frames,
                      original_size, compressed_size):
    n_i = sum(1 for f in encoded_frames if f['type'] == 'I')
    n_p = sum(1 for f in encoded_frames if f['type'] == 'P')

    ratio = original_size / compressed_size

    print("=" * 50)
    print("  PART 5A — METRICS")
    print("=" * 50)
    print(f"  Frames           : {n_i + n_p}  (I={n_i}  P={n_p})")
    print(f"  Original size    : {original_size:,} bytes")
    print(f"  Compressed size  : {compressed_size:,} bytes")
    print(f"  Compression ratio: {ratio:.2f}x")
    print("=" * 50)

    return dict(ratio=ratio,
                original_size=original_size, compressed_size=compressed_size,
                n_i=n_i, n_p=n_p)


# ---- 5B : Pipeline Visualisation ----

# ax → mise en forme dark theme
def _ax_style(ax, title=""):
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#aaaacc')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    if title:
        ax.set_title(title, color='#ccccff', fontsize=9, pad=4)


# preprocessed, frames_bgr, encoded/decoded_frames, metrics → figure 6 lignes sauvegardée
def visualise_pipeline(preprocessed, frames_bgr, encoded_frames,
                       decoded_frames, metrics,
                       save_path="pipeline_visualisation.png"):
    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor('#0f0f1a')
    plt.suptitle("MPEG-4 Simplified Encoder — Full Pipeline Visualisation",
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    gs = GridSpec(6, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Row 1 : affichage des frames originales
    n_show = min(4, len(frames_bgr))
    for k in range(n_show):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(cv2.cvtColor(frames_bgr[k], cv2.COLOR_BGR2RGB))
        ax.axis('off')
        _ax_style(ax, f"Frame {k}")

    # Row 2 : affichage des canaux Y / Cb / Cr de la frame 0
    Y0, Cb0, Cr0 = preprocessed[0]
    for col, (ch, title, cmap) in enumerate([
        (Y0,  "Y  (Luma)",      'gray'),
        (Cb0, "Cb (Blue diff)", 'Blues'),
        (Cr0, "Cr (Red diff)",  'Reds'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(ch, cmap=cmap, vmin=0, vmax=255)
        ax.axis('off')
        _ax_style(ax, title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3 : pipeline DCT sur un bloc 8×8 central
    h0, w0 = Y0.shape
    by, bx = (h0 // 2) // 8 * 8, (w0 // 2) // 8 * 8
    raw_block = Y0[by:by+8, bx:bx+8].astype(np.float32)
    qm        = get_quant_matrix(QUALITY)
    dct_c     = dctn(raw_block - 128, norm='ortho')
    q_block   = np.round(dct_c / qm)
    recon     = np.clip(idctn(q_block * qm, norm='ortho') + 128, 0, 255)

    # affichage des 4 étapes : brut → DCT → quantisé → reconstruit
    for col, (data, title, cmap) in enumerate([
        (raw_block,       "Raw 8×8 pixels",    'gray'),
        (np.abs(dct_c),   "DCT coefficients",  'hot'),
        (np.abs(q_block), "Quantised (int16)", 'plasma'),
        (recon,           "Reconstructed",      'gray'),
    ]):
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(data, cmap=cmap, interpolation='nearest')
        ax.axis('off')
        _ax_style(ax, title)

    # Row 4 : recherche + affichage des motion vectors sur la première P-frame
    mv_idx = next(
        (i for i, f in enumerate(encoded_frames) if f['type'] == 'P'), None)

    ax_mv = fig.add_subplot(gs[3, :])  # pleine largeur
    ax_mv.set_facecolor('#1a1a2e')
    if mv_idx is not None:
        fd    = encoded_frames[mv_idx]
        Y_ref = decoded_frames[mv_idx][0]
        ax_mv.imshow(Y_ref, cmap='gray', vmin=0, vmax=255, aspect='auto')

        mvs = fd['motion_vectors']
        h_f, w_f = fd['frame_shape']
        BLK = 16
        xs, ys, us, vs = [], [], [], []
        # parcours des blocs → collecte des vecteurs non nuls
        for i, (dy, dx) in enumerate(mvs):
            if abs(dx) > 0 or abs(dy) > 0:
                cx = (i % (w_f // BLK)) * BLK + BLK // 2
                cy = (i // (w_f // BLK)) * BLK + BLK // 2
                xs.append(cx); ys.append(cy)
                us.append(dx); vs.append(dy)
        if xs:
            ax_mv.quiver(xs, ys, us, vs, color='#00ffcc',
                         scale=1, scale_units='xy', angles='xy',
                         width=0.001, alpha=0.75)
        ax_mv.set_xlim(0, w_f); ax_mv.set_ylim(h_f, 0)
        _ax_style(ax_mv, f"Motion Vectors — frame {mv_idx} (P)")
    else:
        ax_mv.text(0.5, 0.5, "No P-frames found",
                   ha='center', va='center', color='white', fontsize=12)
    ax_mv.axis('off')

    # Row 5 : affichage carte résiduelle + frame Y reconstruite
    if mv_idx is not None:
        Y_orig  = preprocessed[mv_idx][0].astype(np.float32)
        Y_recon = decoded_frames[mv_idx][0].astype(np.float32)
        residual_map = np.abs(Y_orig - Y_recon)

        ax_r = fig.add_subplot(gs[4, :2])
        im_r = ax_r.imshow(residual_map, cmap='inferno', vmin=0, vmax=50)
        ax_r.axis('off')
        _ax_style(ax_r, f"Residual map — frame {mv_idx}")
        plt.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)

        ax_k = fig.add_subplot(gs[4, 2:])
        ax_k.imshow(Y_recon, cmap='gray', vmin=0, vmax=255)
        ax_k.axis('off')
        _ax_style(ax_k, f"Reconstructed Y — frame {mv_idx}")

    # Row 6 : affichage comparaison des tailles
    ax_bar = fig.add_subplot(gs[5, :])  # pleine largeur

    sizes = [metrics['original_size']/1024, metrics['compressed_size']/1024]
    bars  = ax_bar.bar(['Original\n(raw YCbCr)', 'Compressed\n(.bin)'],
                       sizes, color=['#ef9a9a', '#80cbc4'],
                       edgecolor='none', width=0.5)
    # parcours des barres → annotation KB
    for b, v in zip(bars, sizes):
        ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + max(sizes)*0.01,
                    f"{v:.0f} KB", ha='center', va='bottom',
                    color='white', fontsize=9, fontweight='bold')
    ax_bar.set_ylabel("Size (KB)", color='#aaaacc', fontsize=8)
    _ax_style(ax_bar, f"Size comparison  (ratio = {metrics['ratio']:.2f}×)")

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"[Part 5] Figure saved → '{save_path}'")


# ---- Bonus : Compression ratio vs Quality Factor ----

# preprocessed → courbe ratio en fonction du quality factor
def plot_ratio_vs_quality(preprocessed,
                          quality_values=None,
                          save_path="ratio_vs_quality.png"):
    import pickle, zlib
    if quality_values is None:
        quality_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    Y0, Cb0, Cr0 = preprocessed[0]
    raw_size = Y0.nbytes + Cb0.nbytes + Cr0.nbytes
    ratios = []

    # parcours des niveaux de qualité → encodage frame 0 + mesure ratio
    for q in quality_values:
        encoded, shapes, qm = encode_iframe(Y0, Cb0, Cr0, quality=q)
        comp_size = len(zlib.compress(pickle.dumps(
            {"encoded": encoded, "shapes": shapes, "qm": qm}), level=6))
        ratios.append(raw_size / comp_size)

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    fig.patch.set_facecolor('#0f0f1a')

    # ax → mise en forme dark theme (locale)
    def _s(ax, title, xlabel, ylabel):
        ax.set_facecolor('#1a1a2e')
        ax.set_title(title,   color='#ccccff', fontsize=11)
        ax.set_xlabel(xlabel, color='#aaaacc', fontsize=9)
        ax.set_ylabel(ylabel, color='#aaaacc', fontsize=9)
        ax.tick_params(colors='#aaaacc')
        for sp in ax.spines.values(): sp.set_edgecolor('#333355')
        ax.grid(color='#222244', linestyle='--', linewidth=0.5)

    ax1.plot(quality_values, ratios, 'o-', color='#00e5ff',
             linewidth=2, markersize=6, markerfacecolor='white')
    _s(ax1, "Compression Ratio vs Quality", "Quality Factor", "Ratio (×)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"[Part 5] Ratio vs Quality chart saved → '{save_path}'")







metrics = evaluate_pipeline(
    preprocessed, decoded_frames, encoded_frames,
    original_size, compressed_size
)

visualise_pipeline(
    preprocessed, frames_bgr, encoded_frames,
    decoded_frames, metrics
)

plot_ratio_vs_quality(preprocessed)
