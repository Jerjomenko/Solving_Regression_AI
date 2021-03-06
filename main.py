from tensorflow.keras import models
from tensorflow.keras.preprocessing.text import Tokenizer
from sys import argv


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



tehnology_list =["3D моделирование", "AR/VR", "BigData", "Аддитивные технологии", "Беспилотники", "Биометрия", "Биотехнологии", "Блокчейн", "Зеленые технологии", "Интернет вещей", "Искусственный интеллект и машинное обучение", "Компьютерное зрение", "Нанотехнологии", "Нейротехнологии", "Новые и портативные источники энергии", "Новые материалы", "Робототехника", None]

unique_frame = df2['Сфера стартапа'].unique()


model = models.load_model("best.hdf5")

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


class ListToUser:

  def __init__(self, x):
    self.x = x

  def parse_df(self):
    spis = []
    spis_sdel = []
    for i in range(len(df2['Сфера стартапа'])):
      if self.x == df2['Сфера стартапа'][i]:
        try:
          spis.append(int(df2['Итоговая сумма, долл'][i]))
          spis_sdel.append(i)
        except:
          pass
      else:
        pass

    list_of = []
    for i in range(5):
      for i in spis_sdel:
        if int(df2['Итоговая сумма, долл'][i]) == max(spis):
          list_of.append(df2.loc[i])
          spis.remove(max(spis))
          break
    return list_of

  def show_keys(self):
    l = self.parse_df()
    output = []
    for i in range(len(l)):
      out = []
      out.append(l[i][47])
      out.append(l[i][48])
      out.append(f"{l[i][3]} {int(l[i][45])}$ {str(l[i][1])}")

      output.append(out)

    return output


def end_predict(tw, p, th, s, tx):
    w = ListToUser(p)
    end_l = w.show_keys()
    x = MyPredict(tw, p, th, s)
    end_p = x.all_together()
    y = PredictText(tx)
    end_p2 = y.input_text_preprocessing()
    out = model.predict([end_p, end_p2])
    return f"При данных парамтрах фирмы предположительные инвестиции могут составить - {int(out)} $." , end_l


try:
    tw, p, th, s, tx = argv
except:
    tw = "Лондон"
    p = "FoodTech"
    th = "Зеленые технологии"
    s = "Идея"
    tx = "Самая лучшая технология в мире, на базе нанотехнологий с использованием регресивного микробиологического подхода гинетики и алгоритмов пятого уровня в ускоренном режиме."

out_to_user = end_predict(tw, p, th, s, tx)
print(out_to_user[0])

for i in out_to_user[1]:
  print(i)