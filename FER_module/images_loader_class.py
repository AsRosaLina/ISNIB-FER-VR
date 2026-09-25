# + multi scale luminance
def load_ferplus_nocache(ferplus_path):
    emotion_folder_map = {
        'angry':    0,
        'contempt': 1,
        'disgust':  2,
        'fear':     3,
        'happy':    4,
        'neutral':  5,
        'sad':      6,
        'suprise':  7,
    }
    emotion_names = {v: k.capitalize() for k, v in emotion_folder_map.items()}

    X, y = {}, {}
    for split_name in ['train', 'validation', 'test']:
        split_path = os.path.join(ferplus_path, split_name)
        if not os.path.isdir(split_path):
            print(f"split path not found: {split_path}")
            X[split_name], y[split_name] = [], np.array([], dtype=np.int32)
            continue

        paths, labels = [], []
        print(f"\n{split_name.upper()} split:")
        for emotion_folder, emotion_id in emotion_folder_map.items():
            emotion_path = os.path.join(split_path, emotion_folder)
            if not os.path.isdir(emotion_path):
                print(f"  {emotion_folder:10}: folder missing, skipping")
                continue
            img_files = [f for f in os.listdir(emotion_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  {emotion_folder:10} ({emotion_names[emotion_id]:10}): {len(img_files):5} images")
            for fname in img_files:
                paths.append(os.path.join(emotion_path, fname))
                labels.append(emotion_id)

        X[split_name] = paths
        y[split_name] = np.array(labels, dtype=np.int32)
        print(f"  total: {len(paths)}  distribution: {np.bincount(y[split_name], minlength=8)}")

    return X, y

def make_multiscale_luminance(img_gray):
    img_f = img_gray.astype(np.float32)

    ch0 = img_f

    blurred = cv2.GaussianBlur(img_f, ksize=(0, 0), sigmaX=15)
    ch1_raw = img_f - blurred          
    ch1 = ch1_raw - ch1_raw.min()
    rng = ch1.max()
    if rng > 1e-6:
        ch1 = ch1 / rng * 255.0
    else:
        ch1 = np.full_like(img_f, 128.0)

    lap = cv2.Laplacian(img_gray, ddepth=cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    if lap_abs.max() > 1e-6:
        ch2 = lap_abs / lap_abs.max() * 255.0
    else:
        ch2 = np.zeros_like(img_f)

    return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)


print("make_multiscale_luminance defined")

def cache_from_raw_ferplus(raw_paths, raw_labels, cache_dir, use_float16=True):
    os.makedirs(cache_dir, exist_ok=True)
    out_paths, out_labels = [], []
    skipped = 0
    total   = len(raw_paths)

    for idx, (img_path, label) in enumerate(zip(raw_paths, raw_labels)):
        if idx % 2000 == 0 and idx > 0:
            print(f"  {idx}/{total}  ({idx/total*100:.1f}%)...")

        img_stem   = os.path.splitext(os.path.basename(img_path))[0]
        folder_tag = os.path.basename(os.path.dirname(img_path))
        cache_path = os.path.join(cache_dir, f"{folder_tag}_{img_stem}.npy")

        if not os.path.exists(cache_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue

            img_rgb_for_align = np.stack([img, img, img], axis=-1)

            try:
                aligned_rgb = face_aligner.process(img_rgb_for_align)
                aligned_gray = aligned_rgb[:, :, 0].astype(np.uint8)

                multiscale = make_multiscale_luminance(aligned_gray)

                if use_float16:
                    multiscale = multiscale.astype(np.float16)
                np.save(cache_path, multiscale)
            except Exception:
                skipped += 1
                continue

        out_paths.append(cache_path)
        out_labels.append(label)

    print(f"  done — cached: {len(out_paths)}  skipped: {skipped}")
    return out_paths, np.array(out_labels, dtype=np.int32)


def load_npy(path_bytes, label):
    def _load(p):
        arr = np.load(p.numpy().decode('utf-8')).astype(np.float32)
        return arr
    img = tf.py_function(_load, [path_bytes], tf.float32)
    img.set_shape([224, 224, 3])
    return img, label


def augment_fn(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=20.0)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def make_dataset(paths, labels, batch_size, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), reshuffle_each_iteration=True)
    ds = ds.map(load_npy, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


print("function definitions ready")
