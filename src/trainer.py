        noise = np.random.normal(0, 1, size=(BATCH_SIZE, 20))
        normal_domains = train[(index * BATCH_SIZE):(index + 1) * BATCH_SIZE]

        generated_domains = genr.predict(noise, verbose=0)

        labels_size = (BATCH_SIZE, 1)

        labels_real = np.random.normal(0, 1, size=labels_size)
        labels_fake = np.zeros(shape=labels_size)

        if index % 2 == 0:
            training_domains = normal_domains
            labels = labels_real
        else:
            training_domains = generated_domains
            labels = labels_fake

        # training discriminator on both Normal and generated domains

        disc.trainable = True
        # disc_history = disc.train_on_batch(training_domains, labels,reset_metrics=True,return_dict=True)
        disc_history1 = disc.train_on_batch(normal_domains, labels_real, reset_metrics=True, return_dict=True)
        disc_history2 = disc.train_on_batch(generated_domains, labels_fake, reset_metrics=True, return_dict=True)
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
