import cv2
import numpy as np
import os
from pathlib import Path
###################################### PART 1 : PREPROCESSIONG ############################################
def load_frames(frames_dir):
    """Load all frames from a directory, sorted by name."""
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))  # loads as BGR
        if img is not None:
            frames.append(img)
    print(f"Loaded {len(frames)} frames")
    return frames

def bgr_to_ycbcr(bgr_frame):
    """Convert a BGR frame to YCbCr with chroma subsampling (4:2:0)."""
    
    # Step 1: BGR -> YCbCr
    ycbcr = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
    
    # cv2 returns in order Y, Cr, Cb — reorder to Y, Cb, Cr
    Y  = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 2]
    Cr = ycbcr[:, :, 1]
    
    # Step 2: Chroma subsampling 4:2:0
    # Cb and Cr are downsampled by 2x in both width and height
    h, w = Y.shape
    Cb_sub = cv2.resize(Cb, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    Cr_sub = cv2.resize(Cr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    
    return Y, Cb_sub, Cr_sub

def ycbcr_to_bgr(Y, Cb_sub, Cr_sub):
    """Reverse the process: YCbCr back to BGR (for verification)."""
    
    h, w = Y.shape
    
    # Upsample Cb and Cr back to full size
    Cb = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Stack in Y, Cr, Cb order (as cv2 expects)
    ycrcb = np.stack([Y, Cr, Cb], axis=2)
    
    # Convert back to BGR
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return bgr


# --- Run it ---
frames_bgr = load_frames("frames")

preprocessed = []
for frame in frames_bgr:
    Y, Cb, Cr = bgr_to_ycbcr(frame)
    preprocessed.append((Y, Cb, Cr))

print(f"Preprocessed {len(preprocessed)} frames")
print(f"Y shape:  {preprocessed[0][0].shape}")   # full resolution
print(f"Cb shape: {preprocessed[0][1].shape}")   # half resolution
print(f"Cr shape: {preprocessed[0][2].shape}")   # half resolution
