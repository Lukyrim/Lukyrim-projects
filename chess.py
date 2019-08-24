import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn import preprocessing

base = pd.read_csv('tic-tac-toe.csv')
previsores = base.iloc[:, 0:9].values
classe = base.iloc[:, 9].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)


previsores_treinamento, previsores_teste, classe_treinamento, classes_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units=5, activation='relu', input_dim=9))
classificador.add(Dense(units=5, activation='relu'))
classificador.add(Dense(units=2, activation='sigmoid'))
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)