def adversarial(g, d):
    """
    Adversarial Model
    :return: Adversarial model
    """
    adv_model = Sequential()
    adv_model.add(g)
    d.trainable = False
    adv_model.add(d)
    return adv_model