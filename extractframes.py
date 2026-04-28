import cv2
import os

video = cv2.VideoCapture("flower_cif.y4m")
os.makedirs("frames2", exist_ok=True)

fps = video.get(cv2.CAP_PROP_FPS)
every_n_seconds = 0.3  # save one frame every 0.5 seconds
every_n_frames = int(fps * every_n_seconds)
#max_frames = 100

i = 0
saved = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    cv2.imwrite(f"frames2/frame_{i:04d}.png", frame)
    cv2.imwrite(f"frames2/frame_{saved:04d}.png", frame)
    saved += 1

#    if saved >= max_frames:
 #       break

    i += 1

video.release()
print(f"Saved {saved} frames")