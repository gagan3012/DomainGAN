        disc_history = np.mean([disc_history1['loss'], disc_history2['loss']])
        disc_acc = np.mean([disc_history1['accuracy'], disc_history2['accuracy']])
        disc_dict = {'loss': disc_history, 'accuracy': disc_acc}
        disc.trainable = False

        noise = np.random.normal(0, 1, size=(BATCH_SIZE, 20))  # random latent vectors.
        misleading_targets = np.random.normal(0, 1, size=labels_size)
        gan_history = gan.train_on_batch(noise, misleading_targets, reset_metrics=True, return_dict=True)
            print({index: gan_history}, {index: disc_dict})
print(gan.summary())
print(genr.summary())
print(disc.summary())
