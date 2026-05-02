import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.fft import dctn, idctn

########################################## PART 5 : EVALUATION & VISUALISATION ##############################################


# ---- 5A : Quality Metrics ----

# original, reconstructed (uint8) → PSNR en dB
def compute_psnr(original, reconstructed):
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


# original, reconstructed (uint8) → SSIM scalaire
def compute_ssim(original, reconstructed):
    orig  = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(orig,  (9, 9), 1.5)
    mu2 = cv2.GaussianBlur(recon, (9, 9), 1.5)

    sigma1_sq = cv2.GaussianBlur(orig  ** 2, (9, 9), 1.5) - mu1 ** 2
    sigma2_sq = cv2.GaussianBlur(recon ** 2, (9, 9), 1.5) - mu2 ** 2
    sigma12   = cv2.GaussianBlur(orig * recon, (9, 9), 1.5) - mu1 * mu2

    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-10)
    return float(np.mean(ssim_map))


# preprocessed, decoded_frames, encoded_frames, tailles → affichage métriques + dict résultats
def evaluate_pipeline(preprocessed, decoded_frames, encoded_frames,
                      original_size, compressed_size):
    n = len(decoded_frames)
    n_i = sum(1 for f in encoded_frames if f['type'] == 'I')
    n_p = sum(1 for f in encoded_frames if f['type'] == 'P')
    n_b = sum(1 for f in encoded_frames if f['type'] == 'B')

    # calcul PSNR et SSIM par frame sur le canal Y uniquement
    psnr_list = [compute_psnr(preprocessed[i][0], decoded_frames[i][0]) for i in range(n)]
    ssim_list = [compute_ssim(preprocessed[i][0], decoded_frames[i][0]) for i in range(n)]

    avg_psnr = np.mean([p for p in psnr_list if not np.isinf(p)])
    avg_ssim = np.mean(ssim_list)
    ratio    = original_size / compressed_size

    print("=" * 50)
    print("  PART 5A — METRICS")
    print("=" * 50)
    print(f"  Frames           : {n}  (I={n_i}  P={n_p}  B={n_b})")
    print(f"  Original size    : {original_size:,} bytes")
    print(f"  Compressed size  : {compressed_size:,} bytes")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Average PSNR     : {avg_psnr:.2f} dB")
    print(f"  Average SSIM     : {avg_ssim:.4f}")
    print("=" * 50)

    return dict(psnr=psnr_list, ssim=ssim_list, avg_psnr=avg_psnr,
                avg_ssim=avg_ssim, ratio=ratio,
                original_size=original_size, compressed_size=compressed_size,
                n_i=n_i, n_p=n_p, n_b=n_b)


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
                       decoded_frames, reconstructed_refs, metrics,
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

    # Row 4 : affichage des motion vectors sur la première frame P ou B
    mv_idx = next(
        (i for i, f in enumerate(encoded_frames) if f['type'] in ('P', 'B')), None)

    ax_mv = fig.add_subplot(gs[3, :])  # pleine largeur
    ax_mv.set_facecolor('#1a1a2e')
    if mv_idx is not None:
        fd    = encoded_frames[mv_idx]
        ftype = fd['type']
        Y_ref = decoded_frames[mv_idx][0]
        ax_mv.imshow(Y_ref, cmap='gray', vmin=0, vmax=255, aspect='auto')

        mvs = fd['motion_vectors'] if ftype == 'P' else fd['mv_forward']
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
        _ax_style(ax_mv, f"Motion Vectors — frame {mv_idx} ({ftype})")
    else:
        ax_mv.text(0.5, 0.5, "No P/B-frames found",
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

    # Row 6 : affichage PSNR par frame + comparaison des tailles
    ax_psnr = fig.add_subplot(gs[5, :2])
    ax_bar  = fig.add_subplot(gs[5, 2:])

    colour_map  = {'I': '#4fc3f7', 'P': '#81c784', 'B': '#ffb74d'}
    bar_colours = [colour_map[f['type']] for f in encoded_frames]
    finite_psnr = [p if not np.isinf(p) else 60 for p in metrics['psnr']]
    ax_psnr.bar(range(len(finite_psnr)), finite_psnr,
                color=bar_colours, edgecolor='none', width=0.9)
    ax_psnr.axhline(metrics['avg_psnr'], color='white', linestyle='--', linewidth=0.8)
    ax_psnr.set_xlabel("Frame index", color='#aaaacc', fontsize=8)
    ax_psnr.set_ylabel("PSNR (dB)",   color='#aaaacc', fontsize=8)
    _ax_style(ax_psnr, f"PSNR per frame  (avg = {metrics['avg_psnr']:.1f} dB)")
    ax_psnr.legend(handles=[mpatches.Patch(color=c, label=t)
                             for t, c in colour_map.items()],
                   fontsize=7, facecolor='#1a1a2e',
                   labelcolor='white', edgecolor='#333355')

    sizes = [metrics['original_size']/1024, metrics['compressed_size']/1024]
    bars  = ax_bar.bar(['Original\n(raw YCbCr)', 'Compressed\n(.bin)'],
                       sizes, color=['#ef9a9a', '#80cbc4'],
                       edgecolor='none', width=0.5)
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


# ----Compression ratio vs Quality Factor ----

# preprocessed → courbes ratio / PSNR en fonction du quality factor
def plot_ratio_vs_quality(preprocessed,
                          quality_values=None,
                          save_path="ratio_vs_quality.png"):
    import pickle, zlib
    if quality_values is None:
        quality_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    Y0, Cb0, Cr0 = preprocessed[0]
    raw_size = Y0.nbytes + Cb0.nbytes + Cr0.nbytes
    ratios, psnrs = [], []

    # parcours des niveaux de qualité → encodage frame 0 + mesure ratio et PSNR
    for q in quality_values:
        encoded, shapes, qm = encode_iframe(Y0, Cb0, Cr0, quality=q)
        comp_size = len(zlib.compress(pickle.dumps(
            {"encoded": encoded, "shapes": shapes, "qm": qm}), level=6))
        ratios.append(raw_size / comp_size)
        Y_dec, _, _ = decode_iframe(encoded, shapes, qm)
        psnrs.append(compute_psnr(Y0, Y_dec))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f0f1a')

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

    ax2.plot(quality_values, psnrs,  's-', color='#69f0ae',
             linewidth=2, markersize=6, markerfacecolor='white')
    _s(ax2, "PSNR vs Quality", "Quality Factor", "PSNR (dB)")

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
    decoded_frames, reconstructed_refs, metrics
)

plot_ratio_vs_quality(preprocessed)
