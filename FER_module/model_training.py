model.lambda_div = 0.05
 
new_pos_enc = tf.keras.initializers.TruncatedNormal(stddev=0.005)(shape=(1, 196, 256))
model.pos_enc.assign(new_pos_enc)
model.pos_enc.trainable = True
print(f"pos_enc reinitialized  shape={model.pos_enc.shape}  norm={tf.norm(model.pos_enc).numpy():.4f}")
 
steps_per_epoch_p2 = len(ferplus_tr_cached) // batch_size
total_steps_p2     = 100 * steps_per_epoch_p2
warmup_steps_p2    = 4 * steps_per_epoch_p2
 
lr_schedule_p2 = WarmUpCosine(
    base_lr=1e-4,
    total_steps=total_steps_p2,
    warmup_steps=warmup_steps_p2,
    alpha=0.02
)
 
with strategy.scope():
    model.backbone.trainable = True
    for layer in model.backbone.layers[:100]:
        layer.trainable = False
    model.patch_dw.trainable    = True
    model.patch_up.trainable    = True
    model.patch_norm.trainable  = True
    model.cross_attn.trainable  = True
    model.query_comm.trainable  = True
    model.wq_pool.trainable     = True
    model.fc1.trainable         = True
    model.emotion_out.trainable = True
    model.lambda_div = 0.25
 
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule_p2,
            weight_decay=2e-4,
            global_clipnorm=2.0),
        loss={'emotion_output': SmoothedSparseCE(num_classes=8, smoothing=0.1)},
        metrics={'emotion_output': ['accuracy']}
    )
 
trainable     = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
print(f"trainable : {trainable:,}  non-trainable : {non_trainable:,}")
print(f"blend_alpha at start: {model.get_blend_alpha():.3f}")
 
es_p2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', mode='max',
    patience=20, min_delta=0.0002,
    restore_best_weights=True, verbose=1)
 
ckpt_p2 = tf.keras.callbacks.ModelCheckpoint(
    filepath='/kaggle/working/_best.weights.h5',
    monitor='val_accuracy', mode='max',
    save_best_only=True, save_weights_only=True, verbose=1)
 
tb_p2 = tf.keras.callbacks.TensorBoard(
    log_dir=f"/kaggle/working/logs_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    histogram_freq=0)
 
class DivLossLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tl = logs.get('task_loss', float('nan'))
        dl = logs.get('div_loss',  float('nan'))
        if not np.isnan(tl):
            print(f"  task_loss: {tl:.4f}  div_loss: {dl:.4f}  "
                  f"weighted_div: {self.model.lambda_div * dl:.4f}")
        alpha_val = self.model.get_blend_alpha()
        print(f"  blend_alpha: {alpha_val:.3f}  "
              f"({'entropy-dominant' if alpha_val > 0.6 else 'gate-dominant' if alpha_val < 0.4 else 'balanced'})")
 
print(f"\n train: {len(ferplus_tr_cached):,}  val: {len(ferplus_val_cached):,}")
 
history_p2 = model.fit(
    train_ds_ferplus,
    epochs=finetune_epoch,
    validation_data=val_ds_ferplus,
    callbacks=[es_p2, ckpt_p2, tb_p2, DivLossLogger()],
    verbose=1
)
 
p2_best_val_acc = max(history_p2.history['val_accuracy'])
p2_epochs_run   = len(history_p2.history['loss'])
print(f"\ncomplete — epochs: {p2_epochs_run}  best val acc: {p2_best_val_acc:.4f}")
print(f"final blend_alpha : {model.get_blend_alpha():.3f}")
 
os.makedirs('/kaggle/working/', exist_ok=True)
with open('/kaggle/working/history_p2.json', 'w') as f:
    json.dump(history_p2.history, f)
model.save_weights('/kaggle/working/_final.weights.h5')
model.save('/kaggle/working/_model.keras')
 
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(history_p2.history['accuracy'],     label='train')
axes[0].plot(history_p2.history['val_accuracy'], '--', label='val')
axes[0].set_title('accuracy'); axes[0].legend(); axes[0].grid()
 
axes[1].plot(history_p2.history['loss'],     label='train')
axes[1].plot(history_p2.history['val_loss'], '--', label='val')
axes[1].set_title('total loss'); axes[1].legend(); axes[1].grid()
 
if 'task_loss' in history_p2.history:
    axes[2].plot(history_p2.history['task_loss'], label='task loss')
    axes[2].plot(
        [v * model.lambda_div for v in history_p2.history['div_loss']],
        label=f'div loss x{model.lambda_div}')
    axes[2].set_title('task vs diversity loss'); axes[2].legend(); axes[2].grid()
else:
    axes[2].set_visible(False)
 
plt.tight_layout()
plt.savefig('/kaggle/working/urves.png')
plt.show()
print("curves saved")
