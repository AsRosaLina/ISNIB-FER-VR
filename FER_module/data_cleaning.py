import shutil, hashlib, random
from pathlib import Path
from collections import defaultdict

FERPLUS_RAW = Path('/kaggle/input/datasets/arnabkumarroy02/ferplus')
OUTPUT_ROOT = Path('/kaggle/working/ferplus_cleaned')

SPLITS_TO_MERGE = ['train', 'validation', 'test']
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
SEED        = 42
IMAGE_EXTS  = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

random.seed(SEED)

def get_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def is_image(path):
    return path.suffix.lower() in IMAGE_EXTS

print("=" * 70)
print("STEP 1 - Scanning")
print("=" * 70)

all_images   = defaultdict(list)
split_counts = {}

for split in SPLITS_TO_MERGE:
    split_dir = FERPLUS_RAW / split
    if not split_dir.exists():
        print(f"  [SKIP] {split} not found")
        continue
    split_counts[split] = {}
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        imgs = [p for p in class_dir.iterdir() if is_image(p)]
        all_images[class_dir.name].extend(imgs)
        split_counts[split][class_dir.name] = len(imgs)

classes_sorted = sorted(all_images.keys())
total_before   = sum(len(v) for v in all_images.values())

print(f"\n  {'Class':<14}", end='')
for s in SPLITS_TO_MERGE:
    if s in split_counts:
        print(f"  {s:>12}", end='')
print(f"  {'TOTAL':>8}")

for cls in classes_sorted:
    print(f"  {cls:<14}", end='')
    row_total = 0
    for s in SPLITS_TO_MERGE:
        n = split_counts.get(s, {}).get(cls, 0)
        print(f"  {n:>12,}", end='')
        row_total += n
    print(f"  {row_total:>8,}")

print(f"\n  Total found: {total_before:,}")

print("\n" + "=" * 70)
print("STEP 2 - Deduplication (MD5 only)")
print("=" * 70)

clean_images  = {}
total_removed = 0

for cls in classes_sorted:
    paths    = all_images[cls]
    seen_md5 = set()
    kept     = []
    removed  = 0

    for p in paths:
        m = get_md5(p)
        if m in seen_md5:
            removed += 1
            continue
        seen_md5.add(m)
        kept.append(p)

    clean_images[cls] = kept
    total_removed += removed
    print(f"  {cls:<14}  before: {len(paths):>6,}  removed: {removed:>5,}  kept: {len(kept):>6,}")

total_after = sum(len(v) for v in clean_images.values())
print(f"\n  Duplicates removed : {total_removed:,}")
print(f"  Clean total        : {total_after:,}")

print("\n" + "=" * 70)
print("STEP 3 - Resplitting  (80% train / 10% val / 10% test)")
print("=" * 70)

train_split = {}
val_split   = {}
test_split  = {}

for cls in classes_sorted:
    imgs    = clean_images[cls].copy()
    random.shuffle(imgs)
    n       = len(imgs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    train_split[cls] = imgs[:n_train]
    val_split[cls]   = imgs[n_train : n_train + n_val]
    test_split[cls]  = imgs[n_train + n_val :]
    print(f"  {cls:<14}  train: {len(train_split[cls]):>6,}  val: {len(val_split[cls]):>5,}  test: {len(test_split[cls]):>5,}")

print("\n" + "=" * 70)
print(f"STEP 4 - Writing to {OUTPUT_ROOT}")
print("=" * 70)

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

for split_name, split_data in [('train', train_split), ('validation', val_split), ('test', test_split)]:
    for cls, paths in split_data.items():
        dest_dir = OUTPUT_ROOT / split_name / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            shutil.copy2(src, dest_dir / src.name)
    n = sum(len(v) for v in split_data.values())
    print(f"  {split_name:<12}  {n:,} images written")

print("\n" + "=" * 70)
print("FINAL AUDIT")
print("=" * 70)
print(f"\n  {'Class':<14}  {'Train':>8}  {'Val':>7}  {'Test':>7}  {'Total':>8}  {'Val%':>5}  {'Test%':>6}")
print("  " + "-" * 65)

grand_train = grand_val = grand_test = 0
for cls in classes_sorted:
    tr  = len(train_split[cls])
    va  = len(val_split[cls])
    te  = len(test_split[cls])
    tot = tr + va + te
    print(f"  {cls:<14}  {tr:>8,}  {va:>7,}  {te:>7,}  {tot:>8,}  "
          f"{va/tot*100:>4.1f}%  {te/tot*100:>5.1f}%")
    grand_train += tr
    grand_val   += va
    grand_test  += te

grand_total = grand_train + grand_val + grand_test
print("  " + "-" * 65)
print(f"  {'TOTAL':<14}  {grand_train:>8,}  {grand_val:>7,}  {grand_test:>7,}  {grand_total:>8,}  "
      f"{grand_val/grand_total*100:>4.1f}%  {grand_test/grand_total*100:>5.1f}%")

print(f"\n  Before cleaning    : {total_before:,}")
print(f"  Duplicates removed : {total_removed:,}")
print(f"  After cleaning     : {total_after:,}")
print(f"  Ready at           : {OUTPUT_ROOT}")
print("=" * 70)
print("DONE")
print("=" * 70)
