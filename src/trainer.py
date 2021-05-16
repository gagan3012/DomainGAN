
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, 20))  # random latent vectors.
        misleading_targets = np.random.normal(0, 1, size=labels_size)
        gan_history = gan.train_on_batch(noise, misleading_targets, reset_metrics=True, return_dict=True)
            print({index: gan_history}, {index: disc_dict})
print(gan.summary())
print(genr.summary())
print(disc.summary())
