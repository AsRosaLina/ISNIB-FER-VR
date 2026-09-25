import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight

def limit_per_class(paths, labels, max_per_class=7000, num_classes=8, seed=42):
    rng    = np.random.default_rng(seed)
    paths  = np.array(paths)
    labels = np.array(labels)
    selected_indices = []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, max_per_class, replace=False)
        selected_indices.append(cls_idx)
    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)
    return paths[selected_indices].tolist(), labels[selected_indices].tolist()

print("loading ferplus dataset")
print("=" * 80)

X_ferplus, y_ferplus = load_ferplus_nocache(FERPLUS_PATH)

ferplus_tr_paths   = X_ferplus['train']
ferplus_val_paths  = X_ferplus['validation']
ferplus_test_paths = X_ferplus['test']

y_tr_raw   = y_ferplus['train']
y_val_raw  = y_ferplus['validation']
y_test_raw = y_ferplus['test']

ferplus_tr_paths, y_tr_raw = limit_per_class(
    ferplus_tr_paths, y_tr_raw, max_per_class=7000
)

print("\ncaching train...")
ferplus_tr_cached, y_tr_cached = cache_from_raw_ferplus(
    ferplus_tr_paths, y_tr_raw,
    cache_dir=os.path.join(FERPLUS_CACHE, 'train'),
    use_float16=True
)

print("\ncaching val...")
ferplus_val_cached, y_val_cached = cache_from_raw_ferplus(
    ferplus_val_paths, y_val_raw,
    cache_dir=os.path.join(FERPLUS_CACHE, 'val'),
    use_float16=True
)

cw_ferplus = compute_class_weight(
    'balanced',
    classes=np.unique(y_tr_cached),
    y=y_tr_cached
)
cw_ferplus = dict(enumerate(cw_ferplus))

train_ds_ferplus = make_dataset(
    ferplus_tr_cached, y_tr_cached,
    batch_size, shuffle=True, augment=True
)

val_ds_ferplus = make_dataset(
    ferplus_val_cached, y_val_cached,
    batch_size, shuffle=False, augment=False
)

emotion_names_ferplus = [
    'Angry', 'Contempt', 'Disgust', 'Fear',
    'Happy', 'Neutral', 'Sad', 'Suprise'
]

counts = np.bincount(y_tr_cached, minlength=8)

print("\n" + "=" * 80)
print("ferplus dataset summary")
print("=" * 80)
print(f"train: {len(ferplus_tr_cached)}  | per-class: {counts}")
print(f"val:   {len(ferplus_val_cached)}  | per-class: {np.bincount(y_val_cached, minlength=8)}")
print(f"test:  {len(ferplus_test_paths)}  | not cached yet")
print(f"class weights: {cw_ferplus}")
print(f"smallest class ({emotion_names_ferplus[np.argmin(counts)]}): {counts.min()}")
print(f"largest  class ({emotion_names_ferplus[np.argmax(counts)]}): {counts.max()}")
print(f"imbalance ratio: {counts.max()/counts.min():.2f}x")
print("=" * 80)
