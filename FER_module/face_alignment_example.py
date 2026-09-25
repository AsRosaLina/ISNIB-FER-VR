import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

image_paths = [
    'angry_face.jpg',
    'fear_glasses_face.jpg',
    'happy_face.jpg',
    'neutral_face.jpg',
    'sad_face.jpg',
    'surprise_glasses_beard_face.jpg',
    'happy_tilted.jpg',
]

for img_path in image_paths:
    label = os.path.splitext(os.path.basename(img_path))[0]

    bgr          = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w         = bgr.shape[:2]

  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=original_rgb)
    det      = face_aligner.face_landmarker.detect(mp_image)
    has_face = bool(det.face_landmarks)

    if has_face:
        lms       = det.face_landmarks[0]
        left_eye  = face_aligner._eye_center(lms, face_aligner.left_eye_idx)
        right_eye = face_aligner._eye_center(lms, face_aligner.right_eye_idx)

        le_px = (left_eye[0]  * w, left_eye[1]  * h)
        re_px = (right_eye[0] * w, right_eye[1] * h)

        angle = np.degrees(np.arctan2(re_px[1] - le_px[1], re_px[0] - le_px[0]))
        ctr   = ((le_px[0] + re_px[0]) / 2, (le_px[1] + re_px[1]) / 2)

        M       = cv2.getRotationMatrix2D(ctr, angle, 1.0)
        rotated = cv2.warpAffine(original_rgb, M, (w, h))

        lm_arr = np.array([[lm.x * w, lm.y * h] for lm in lms],
                           dtype=np.float32).reshape(-1, 1, 2)
        lm_rot = cv2.transform(lm_arr, M).reshape(-1, 2)

        x1 = max(0, int(lm_rot[:, 0].min()));  x2 = min(w, int(lm_rot[:, 0].max()))
        y1 = max(0, int(lm_rot[:, 1].min()));  y2 = min(h, int(lm_rot[:, 1].max()))
        px = int((x2 - x1) * 0.15);            py = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
        x2 = min(w, x2 + px);  y2 = min(h, y2 + py)

        crop        = rotated[y1:y2, x1:x2]
        aligned_224 = cv2.resize(crop, (224, 224))
        lm_xy       = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    else:
        rotated     = original_rgb.copy()
        aligned_224 = cv2.resize(original_rgb, (224, 224))
        lm_xy       = []
        le_px = re_px = None
        angle = 0.0
        x1 = y1 = x2 = y2 = None

    print(f"{label}: face={'YES' if has_face else 'NO'}  roll={angle:.1f}°  landmarks={len(lm_xy)}")

    # figure: 5 panels
    fig = plt.figure(figsize=(23, 5))
    fig.suptitle(f"{label}  |  face aligner pipeline", fontsize=13, y=1.02)
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.06)

    # panel 0 — original
    ax = fig.add_subplot(gs[0])
    ax.imshow(original_rgb)
    ax.set_title(f"① original\n{w}×{h}px", fontsize=10)
    ax.axis('off')

    # panel 1 — landmarks on original
    ax = fig.add_subplot(gs[1])
    ax.imshow(original_rgb)
    if has_face:
        xs = [p[0] for p in lm_xy];  ys = [p[1] for p in lm_xy]
        ax.scatter(xs, ys, s=2, c='lime', linewidths=0)
        ax.scatter(*le_px, s=90, c='cyan',   zorder=5, label='left eye')
        ax.scatter(*re_px, s=90, c='yellow', zorder=5, label='right eye')
        ax.plot([le_px[0], re_px[0]], [le_px[1], re_px[1]],
                c='white', lw=1.5, ls='--')
        ax.legend(fontsize=7, loc='lower right',
                  facecolor='#111', labelcolor='white', markerscale=1.5)
    ax.set_title(f"② 478 landmarks\nroll = {angle:.1f}°", fontsize=10)
    ax.axis('off')

    # panel 2 — roll-corrected + bounding box
    ax = fig.add_subplot(gs[2])
    ax.imshow(rotated)
    if has_face and x1 is not None:
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    ax.set_title("③ roll corrected\n+ face bbox (red)", fontsize=10)
    ax.axis('off')

    # panel 3 — crop only
    ax = fig.add_subplot(gs[3])
    if has_face and x1 is not None:
        crop_show = rotated[y1:y2, x1:x2]
        ax.imshow(crop_show)
        ax.set_title(f"④ face crop\n{crop_show.shape[1]}×{crop_show.shape[0]}px", fontsize=10)
    else:
        ax.imshow(rotated)
        ax.set_title("④ face crop\n(fallback: full frame)", fontsize=10)
    ax.axis('off')

    # panel 4 — final 224×224
    ax = fig.add_subplot(gs[4])
    ax.imshow(aligned_224)
    ax.set_title("⑤ model input\n224 × 224", fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'/kaggle/working/aligner_{label}.png', bbox_inches='tight', dpi=130)
    plt.show()
