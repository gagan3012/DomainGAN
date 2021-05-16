from sklearn.model_selection import train_test_split

from src.decoder import decoder_model
from src.encoder import encoder_model

EPOCH = 5
n=20
e = encoder_model()
d = decoder_model()

adv_model = Sequential()
adv_model.add(e)
adv_model.add(d)
print(adv_model.summary())

train,test = train_test_split(df, test_size=0.1)
train_new,test_new = tokenize(train), tokenize(test)
train_new, test_new= data_dict['X_train'],data_dict['X_test']
print(train_new.shape,test_new.shape)


adv_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
adv_model.fit(train_new, train_new,
             verbose=1,
             validation_data = (test_new, test_new),
             batch_size=128,
             epochs=EPOCH)
loss, accuracy = adv_model.evaluate(test_new, test_new, verbose=1)
print("Loss:",loss,"Accuracy:",accuracy)

model_name = "Autoencodermodel"
MODEL_HOME = "../model/GAN_Models/"
save_model(adv_model,MODEL_HOME + model_name + ".json", MODEL_HOME + model_name + ".h5")

print("testing")
predictions = adv_model.predict(test_new, verbose=1)
sampled = []
for x in predictions:
    word = []
    for y in x:
        word.append(__np_sample(y))
    sampled.append(word)

print("results")
readable = __to_readable_domain(np.array(sampled), inv_map=data_dict['inv_map'])
dfa= df['url'].tolist()
