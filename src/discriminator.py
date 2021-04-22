def discriminator_model():
    """
    Discriminator model:
    :return: Discriminator model
    """
    model = Sequential()
    model.add(encoder_model())
    model.add(Dense(1, activation='relu'))
    model.add(Activation('relu'))
    return model
