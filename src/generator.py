def generator_model():
    """
    Generator model:
    param: noise vector
    :return: generator model
    """
    model = Sequential()
    model.add(Input(shape=(20,)))
    model.add(Dense(480, activation='relu'))
    model.add(ReLU())
    model.add(decoder_model(480))
    return model

