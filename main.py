import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.models import  Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np


df2 = pd.read_excel("Венчурные сделки 2017-2021.xlsx")

# predobrabotka dannih
def getXtrainText(val):
  xTrainText = []
  for i in val:
    xTrainText.append(i)
  xTrainText = np.array(xTrainText)
  return xTrainText

xTrainText = getXtrainText(df2['Краткое описание'])

maxWordsCount = 5000

tokenizer = Tokenizer(num_words=maxWordsCount, filters='!"#$%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0', lower=True, split=' ', oov_token='unknown', char_level=False)

tokenizer.fit_on_texts(xTrainText)
items = list(tokenizer.word_index.items())

xTrainTextIndexes = tokenizer.texts_to_sequences(xTrainText)
xTrainText01 = tokenizer.sequences_to_matrix(xTrainTextIndexes)

# funkcija dlja yTrain
def getYTrain(val):
  yTrain = []
  for i in val:
    if pd.isna(i):
      yTrain.append(int(30000))
    elif (type(i) == float)  and (int(i) > 10000000) or (type(i) == int) and (int(i) > 10000000):
      yTrain.append(int(10000000))
    elif type(i) == float or type(i) == int and i > 1:
      yTrain.append(int(i))
    else:
      yTrain.append(int(31000))
  yTrain = np.array(yTrain)
  return yTrain


tehnology_list =["3D моделирование", "AR/VR", "BigData", "Аддитивные технологии", "Беспилотники", "Биометрия", "Биотехнологии", "Блокчейн", "Зеленые технологии", "Интернет вещей", "Искусственный интеллект и машинное обучение", "Компьютерное зрение", "Нанотехнологии", "Нейротехнологии", "Новые и портативные источники энергии", "Новые материалы", "Робототехника", None]

unique_frame = df2['Сфера стартапа'].unique()


def getParametrProfile(arg):
  xTrain = []
  y = list(unique_frame)
  out = str(0)*len(y)
  out = [int(x) for x in out]
  for i in range(len(arg)):
    if arg[i] in y:
      out[y.index(arg[i])] = 1
      xTrain.append(out)
      out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  return xTrain

def getParametrTehnology(arg):
  x = []
  out = str(0)*len(tehnology_list)
  out = [int(x) for x in out]
  for i in range(len(arg)):
    if arg[i] in tehnology_list:
      out[tehnology_list.index(arg[i])] = 1
      x.append(out)
      out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif type(arg[i]) == str  and "," in arg[i]:
      n = arg[i].split(",")
      for j in range(len(n)):
        n[j] = n[j].strip()
        out[tehnology_list.index(n[j])] = 1
      x.append(out)
      out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
      out[17] = 1
      x.append(out)
      out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  return x



def get_stage_train():
  stage_x_train = []
  for i in df2['Стадия стартапа']:
    if i == "Идея":
      stage_x_train.append([1, 0, 0, 0])
    elif i == "Посевная":
      stage_x_train.append([0, 1, 0, 0])
    elif i == "Ранний рост":
      stage_x_train.append([0, 0, 1, 0])
    else:
      stage_x_train.append([0, 0, 0, 1])

  return stage_x_train


stageXTrain = get_stage_train()
profileXTrain = getParametrProfile(df2['Сфера стартапа'])
tehnolpgyXTrain = getParametrTehnology(df2['Технология'])
cityXTrain = [[1, 0] if x == "Москва" else  [0, 1]  for x in df2['Город']]
yTrain = getYTrain(df2['Итоговая сумма, долл'])



def getAllParametrs():
  x = []
  out = []
  for i in range(940):
    out = cityXTrain[i] + profileXTrain[i] + tehnolpgyXTrain[i] + stageXTrain[i]
    x.append(out)
  return x


xTrain = getAllParametrs()
xTrain = np.array(xTrain)


def my_model():
  input1 = Input((xTrain.shape[1]))
  input2 = Input((xTrainText01.shape[1]))

  x1 = BatchNormalization()(input1)
  x1 = Dropout(0.5)(x1)
  x1 = Dense(10, activation="relu")(x1)
  x1 = Dense(1000, activation="relu")(x1)
  x1 = Dense(100, activation="relu")(x1)

  x2 = BatchNormalization()(input2)
  x2 = Dense(10, activation="relu")(input2)
  x2 = Dropout(0.5)(x2)
  x2 = Dense(1000, activation="tanh")(x2)
  x2 = Dense(5, activation="elu")(x2)

  x = concatenate([x1, x2])

  x = Dense(1000, activation="relu")(x)
  x = Dropout(0.5)(x)
  x = Dense(1, activation="relu")(x)

  model = Model((input1, input2), x)

  def on_epoch_end(epoch, logs):
    mae = logs["val_mae"]
    print(f"На эпохе {epoch} средняя ошибка состовляет {mae}")

  output = LambdaCallback(on_epoch_end=on_epoch_end)

  model.compile(optimizer=Adam(lr=1e-3), loss="mse", metrics=["mae"])

  return model


model2 = my_model()

model = load_model("_528.hdf5")


# Класс MyPredict получает наши вводные данные кроме текста описания и возвращает массив который мы скармливаем нейронке на предокт первым параметром.

class MyPredict:

  def __init__(self, town, profile, tehnology, stage_input):
    # Здесь в классе вводные данные определенны инпутами для тестирования, в реале будем передавать в инит три аргумента которые будут прилетать от пользователя.
    self.town = town
    self.profile = profile
    self.tehnology = tehnology
    self.stage_input = stage_input

    self.unique_fr = ['Advertising & Marketing', 'Transport & Logistics', 'LegalTech', 'FoodTech',
                      'FinTech', 'RetailTech', 'Entertainment', 'CleanTech', 'Business Software',
                      'PropTech', 'E-commerce', 'HRTech', 'EdTech', 'Healthcare',
                      'Мedia & Communication', 'Consumer Goods & Services', 'Gaming', 'Travel',
                      'Telecom', 'SportTech', 'Cybersecurity', 'IndustrialTech', 'SafetyTech',
                      'FashionTech', 'RetailTech', 'InsuranceTech', 'AgroTech', 'SpaceTech']

    self.tehnology_list = ["3D моделирование", "AR/VR", "BigData", "Аддитивные технологии", "Беспилотники",
                           "Биометрия", "Биотехнологии", "Блокчейн", "Зеленые технологии", "Интернет вещей",
                           "Искусственный интеллект и машинное обучение", "Компьютерное зрение", "Нанотехнологии",
                           "Нейротехнологии", "Новые и портативные источники энергии", "Новые материалы",
                           "Робототехника", None]

    self.stage_list = ["Идея", "Посевная", "Ранний рост"]

  def town_preprocess(self):
    if self.town.lower() == "москва":
      return [1, 0]
    else:
      return [0, 1]

  def profili_preprocess(self):
    arg = self.profile
    out = str(0) * len(self.unique_fr)
    out = [int(x) for x in out]
    for i in range(len(self.unique_fr)):
      if arg == self.unique_fr[i]:
        out[self.unique_fr.index(arg)] = 1
        xTrain = out
        out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return xTrain

  def getParametrTehnology(self):
    arg = self.tehnology
    out = str(0) * len(self.tehnology_list)
    out = [int(x) for x in out]
    for i in range(len(out)):
      if arg in self.tehnology_list:
        out[self.tehnology_list.index(arg)] = 1
        x = out
        out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      elif type(arg) == str and "," in arg:
        n = arg.split(",")
        for j in range(len(n)):
          n[j] = n[j].strip()
          out[self.tehnology_list.index(n[j])] = 1
          x = out
          out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      else:
        out[17] = 1
        x = out
        out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      return x

  def stage_param(self):
    st = self.stage_input
    for i in self.stage_list:
      if st == i:
        return [1, 0, 0, 0]
      elif st == i:
        return [0, 1, 0, 0]
      elif st == i:
        return [0, 0, 1, 0]
      else:
        return [0, 0, 0, 1]

  def all_together(self):
    l = []
    l.append(self.town_preprocess())
    l.append(self.profili_preprocess())
    l.append(self.getParametrTehnology())
    l.append(self.stage_param())
    xTr = []

    for i in range(len(l)):
      for j in range(len(l[i])):
        xTr.append(l[i][j])

    xTr = np.array(xTr)
    xTr = np.expand_dims(xTr, axis=0)

    return xTr


class PredictText:

  def __init__(self, text):
    self.text = text

  def input_text_preprocessing(self):
    ls = self.text.split()
    ls = [x.replace(".", "").replace("-", "").replace(",","") for x in ls]
    binded = tokenizer.texts_to_sequences(ls)
    res = []
    for i in binded:
      res += i
    res_pred = tokenizer.sequences_to_matrix([res])
    return res_pred


a = MyPredict("Лондон", "FoodTech", "Зеленые технологии", "Идея")
for_proba = a.all_together()

d = PredictText("Самая лучшая технология в мире, на базе нанотехнологий с использованием регресивного микробиологического подхода гинетики и алгоритмов пятого уровня в ускоренном режиме.")
for_proba2 = d.input_text_preprocessing()

proba = model.predict([for_proba, for_proba2])
print(f"При данных парамтрах фирмы предположительные инвестиции могут составить - {int(proba)} $.")

