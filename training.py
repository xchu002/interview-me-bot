import keras
import nltk
import json
import pickle

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy as np

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents["intents"]:
    #append classes if not in classes
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

    #loop thru intent["patterns"]
    for pattern in intent["patterns"]:
        #tokenize patterns
        word_list = nltk.word_tokenize(pattern)

        for word in word_list:
            cleaned_list = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in ignore_letters]
        documents.append((cleaned_list, intent["tag"]))
        words.extend(cleaned_list)

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training_x = []
training_y = []

#loop every sentence in patterns(documents)
for document in documents:
    #create bag of 0s
    bag = [0] * len(words)
    output = [0] * len(classes)
    #if word is in vocab, turn bag value into 1
    for word_patterns in document[0]:
        if word_patterns in words:
            bag[words.index(word_patterns)] = 1
    training_x.append(bag)
    #turn 0 to 1 in output if tag is a certain class 
    output[classes.index(document[1])] = 1
    training_y.append(output)
    # training_y.extend([document[1]])

    

training_x = np.array(training_x)
training_y = np.array(training_y)

model = Sequential()
model.add(Dense(128, input_shape=(len(training_x[0]),),activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(training_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(training_x, training_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)