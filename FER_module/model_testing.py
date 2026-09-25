import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)


model_path = 'model.keras'
with strategy.scope():
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'EmotionModel':          EmotionModel
        }
    )
print(f"model loaded  total params: {model.count_params():,}")
print(f"blend alpha: {model.get_blend_alpha():.3f}")
print(f"lambda div:  {model.lambda_div}")

# keras evaluation
results = model.evaluate(test_ds_ferplus, return_dict=True, verbose=1)
print(f"\ntest loss     : {results['loss']:.4f}")
print(f"test accuracy : {results.get('emotion_output_accuracy', results.get('accuracy', float('nan'))):.4f}")

# predictions
preds_test = model.predict(test_ds_ferplus, verbose=0)
if isinstance(preds_test, dict):
    preds_test = preds_test['emotion_output']
y_pred_test = np.argmax(preds_test, axis=1)

# true labels
y_true = np.concatenate([y for _, y in test_ds_ferplus], axis=0)

# sklearn metrics
acc_test         = accuracy_score(y_true, y_pred_test)
f1_macro_test    = f1_score(y_true, y_pred_test, average='macro',    zero_division=0)
f1_weighted_test = f1_score(y_true, y_pred_test, average='weighted', zero_division=0)

print(f"\nsklearn accuracy   : {acc_test:.4f}")
print(f"f1 macro           : {f1_macro_test:.4f}")
print(f"f1 weighted        : {f1_weighted_test:.4f}")

cm_test = confusion_matrix(y_true, y_pred_test)
print("\nconfusion matrix:")
print(cm_test)
print("\nclassification report:")
print(classification_report(y_true, y_pred_test, target_names=ferplus_names, zero_division=0))

# raw confusion matrix
fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_test,
    annot=True, fmt='d', cmap='Blues',
    xticklabels=ferplus_names,
    yticklabels=ferplus_names,
    ax=ax_cm
)
ax_cm.set_title(f'phase 1 confusion matrix  acc: {acc_test:.4f}  f1 macro: {f1_macro_test:.4f}')
ax_cm.set_xlabel('predicted')
ax_cm.set_ylabel('true')
plt.tight_layout()
plt.savefig('/kaggle/working/cm_phase1.png', dpi=300, bbox_inches='tight')
plt.show()

# normalized confusion matrix
cm_norm = np.divide(
    cm_test.astype(float),
    cm_test.sum(axis=1, keepdims=True),
    where=cm_test.sum(axis=1, keepdims=True) != 0
)
fig_cm_norm, ax_cm_norm = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_norm,
    annot=True, fmt='.2f', cmap='Blues',
    xticklabels=ferplus_names,
    yticklabels=ferplus_names,
    ax=ax_cm_norm
)
ax_cm_norm.set_title(f'phase 1 normalized confusion matrix  acc: {acc_test:.4f}')
ax_cm_norm.set_xlabel('predicted')
ax_cm_norm.set_ylabel('true')
plt.tight_layout()
plt.savefig('/kaggle/working/cm_phase1_normalized.png', dpi=300, bbox_inches='tight')
plt.show()

print("saved cm_phase1.png and cm_phase1_normalized.png")
