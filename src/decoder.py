def decoder_model(latent_vector=160):
    cnn_filters = [256, 256, 256, 32, 39]
    cnn_kernels = [2, 3, 4, 3, 3]
    cnn_strides = [1, 1, 1, 1, 1]
    dec_convs = []
    dece = int(latent_vector / 20)
    word_index = 20

    inputs = Input(shape=(latent_vector), name="Decoder_Input")
    decoder = Reshape([word_index, dece], input_shape=(latent_vector,))(inputs)
    for i in range(3):
        conv = Conv1D(cnn_filters[i],
                      cnn_kernels[i],
                      padding='same',
                      strides=cnn_strides[i],
                      name='dec_conv%s' % i)(decoder)
        conv = ReLU()(conv)
        dec_convs.append(conv)

    decoder = concatenate(dec_convs)
    decoder = Conv1D(cnn_filters[3],
                     cnn_kernels[3],
                     padding='same',
                     strides=cnn_strides[3],
                     name='dec_conv%s' % 3)(decoder)
    decoder = ReLU()(decoder)
    decoder = Conv1D(cnn_filters[4],
                     cnn_kernels[4],
                     padding='same',
                     strides=cnn_strides[4],
                     name='dec_conv%s' % 4)(decoder)
    decoder = Softmax()(decoder)
    # decoder = Flatten()(decoder)
    # decoder = Dense(word_index)(decoder)
    model = Model(inputs=inputs, outputs=decoder, name='Decoder')
    return model

