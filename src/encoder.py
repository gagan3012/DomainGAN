def encoder_model():
    cnn_filters = [256, 256, 256, 8]
    cnn_kernels = [2, 3, 4, 2]
    cnn_strides = [1, 1, 1, 1]
    en_convs = []

    inputs = Input(shape=(20, 39,), name="Encoder_Input")
    # encoder = Embedding(1000, 39,input_length=20)(inputs)
    for i in range(3):
        conv = Conv1D(cnn_filters[i],
                      cnn_kernels[i],
                      padding='same',
                      strides=cnn_strides[i],
                      name='en_conv%s' % i)(inputs)
        conv = ReLU()(conv)
        en_convs.append(conv)

    encoder = concatenate(en_convs)
    encoder = Conv1D(cnn_filters[3],
                     cnn_kernels[3],
                     padding='same',
                     strides=cnn_strides[3],
                     name='en_conv%s' % 3)(encoder)
    encoder = ReLU()(encoder)
    encoder = Flatten()(encoder)

    model = Model(inputs=inputs, outputs=encoder, name='Encoder')
    return model
