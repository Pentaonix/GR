import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import csv
import statistics

# read data
df = pd.read_csv('AT_final.csv', index_col = None, header = 0)

label = df.loc[:]['#Project']

time = 75
x = df.loc[df['Actual time complete(day)'] == time]['XGBoost']
xgboost = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['LightGBM']
lightgbm = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['LSTM']
lstm = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['PV1']
pv1 = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['PV2']
pv2 = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['PV3']
pv3 = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['ED1']
ed1 = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['ED2']
ed2 = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['ED3']
ed3 = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['ES1']
es1 = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['ES2']
es2 = list(x)

x = df.loc[df['Actual time complete(day)'] == time]['ES3']
es3 = list(x)

#data processing
label = list(dict.fromkeys(label))

for i in range(0,11):
    label[i] = label[i].replace('test-','')

#convert to float form in str form
for i in range(0,10):
    xgboost[i] = xgboost[i].replace(',','.')
    lightgbm[i] = lightgbm[i].replace(',','.')
    lstm[i] = lstm[i].replace(',','.')
    pv1[i] = pv1[i].replace(',','.')
    ed1[i] = ed1[i].replace(',','.')
    es1[i] = es1[i].replace(',','.')
    pv2[i] = pv2[i].replace(',','.')
    ed2[i] = ed2[i].replace(',','.')
    es2[i] = es2[i].replace(',','.')
    pv3[i] = pv3[i].replace(',','.')
    ed3[i] = ed3[i].replace(',','.')
    es3[i] = es3[i].replace(',','.')

#convert to float
for i in range(0,10):
    xgboost[i] = abs(float(xgboost[i]))
    lightgbm[i] = abs(float(lightgbm[i]))
    lstm[i] = abs(float(lstm[i]))
    pv1[i] = abs(float(pv1[i]))
    es1[i] = abs(float(es1[i]))
    ed1[i] = abs(float(ed1[i]))
    pv2[i] = abs(float(pv2[i]))
    es2[i] = abs(float(es2[i]))
    ed2[i] = abs(float(ed2[i]))
    pv3[i] = abs(float(pv3[i]))
    es3[i] = abs(float(es3[i]))
    ed3[i] = abs(float(ed3[i]))

#get means
m_xgboost = round(float(sum(xgboost))/max(len(xgboost),1),2)
m_lstm = round(float(sum(lstm))/max(len(lstm),1),2)
m_lightgbm = round(float(sum(lightgbm))/max(len(lightgbm),1),2)
m_pv1 = round(float(sum(pv1))/max(len(pv1),1),2)
m_ed1 = round(float(sum(ed1))/max(len(ed1),1),2)
m_es1 = round(float(sum(es1))/max(len(es1),1),2)
m_pv2 = round(float(sum(pv2))/max(len(pv2),1),2)
m_ed2 = round(float(sum(ed2))/max(len(ed2),1),2)
m_es2 = round(float(sum(es2))/max(len(es2),1),2)
m_pv3 = round(float(sum(pv3))/max(len(pv3),1),2)
m_ed3 = round(float(sum(ed3))/max(len(ed3),1),2)
m_es3 = round(float(sum(es3))/max(len(es3),1),2)

#get variance
var_xgboost = statistics.variance(xgboost)
var_lightgbm = statistics.variance(lightgbm)
var_lstm = statistics.variance(lstm)
var_pv1 = statistics.variance(pv1)
var_pv2 = statistics.variance(pv2)
var_pv3 = statistics.variance(pv3)
var_ed1 = statistics.variance(ed1)
var_ed2 = statistics.variance(ed2)
var_ed3 = statistics.variance(ed3)
var_es1 = statistics.variance(es1)
var_es2 = statistics.variance(es2)
var_es3 = statistics.variance(es3)

var = [var_xgboost, var_lightgbm, var_lstm, var_pv1, var_pv2, var_pv3, var_ed1, var_ed2, var_ed3, var_es1, var_es2, var_es3]
min_mape = [min(xgboost), min(lightgbm), min(lstm), min(pv1), min(pv2), min(pv3), min(ed1), min(ed2), min(ed3), min(es1), min(es2), min(es3)]
max_mape = [max(xgboost), max(lightgbm), max(lstm), max(pv1), max(pv2), max(pv3), max(ed1), max(ed2), max(ed3), max(es1), max(es2), max(es3)]
means = [m_xgboost, m_lightgbm, m_lstm, m_pv1, m_pv2, m_pv3, m_ed1, m_ed2, m_ed3, m_es1, m_es2, m_es3]
m_label = ['XGBoost', 'LightGBM', 'LSTM', 'PV1', 'PV2', 'PV3', 'ED1', 'ED2', 'ED3', 'ES1', 'ES2', 'ES3']
print(means)
print(min_mape)
  
# opening the csv file in 'w+' mode
file = open('minmax.csv', 'w+', newline ='')
  
# writing the data into the file
with file:    
    write = csv.writer(file)
    write.writerows(map(lambda x: [x], var))

#barchart
#set limit to vertical axis
figure, axis = plt.subplots()
axis.set_ylim(0,100)

#list from 0-11 -> get name
y_pos = np.arange(len(m_label))

#plot the barchart using 2 list(name,value)
plt.bar(y_pos, means, align='center', alpha=0.5)

#change horizontal category name
plt.xticks(y_pos, m_label)

#labeling for vertical axis
axis.set_xlabel('Model')
axis.set_ylabel('Mape(%)')

#barchart's name
plt.title('Means of models at {}%'.format(time))

#set labels for columns
rects = axis.patches
for rect, label in zip(rects, means):
    height = rect.get_height()
    axis.text(rect.get_x() + rect.get_width() / 2, height , label,
            ha='center', va='bottom')

plt.show()


# test-C2013-15,75,,,,,,,,,,"32,44","24,54","76,99"




# #initiate chart
# figure, axis = plt.subplots()
# axis.set_ylim(0,100)

# #get y_pos from label
# y_pos = np.arange(len(label))
# #plot chart
# plt.plot(y_pos, xgboost, color = 'red', marker = 'x')
# plt.plot(y_pos, lightgbm, color = 'blue', marker = 'x')
# plt.plot(y_pos, lstm, color = 'yellow', marker = 'x')
# plt.plot(y_pos, pv1, color = 'green', marker = 'x')
# plt.plot(y_pos, ed1, color = 'purple', marker = 'x')
# plt.plot(y_pos, es1, color = 'black', marker = 'x')

# #rename y_pos
# plt.xticks(y_pos, label)
# #add ticks
# axis.tick_params('y', color = 'red') 
# #chart's name
# plt.title('Prediction at {}%'.format(time))
# #x_pos name
# axis.set_ylabel('Mape')
# plt.show()

