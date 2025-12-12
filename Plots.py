import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

df = pd.read_csv("WORKDATA.csv", parse_dates=["Zeitpunkt"], low_memory=False)

#print(data.dtypes)

#print(data[data['Zeitpunkt'] == "2017-12-31 17:00:00"])

KlosterstraßeSüd = "15-SP-KLO-S 01.06.2016"
KlosterstraßeNord = "15-SP-KLO-N 01.06.2016"
JannowitzbrückeNord = "02-MI-JAN-N 01.04.2015"
JannowitzbrückeSüd = "02-MI-JAN-S 01.04.2015"
KarlMarxAllee = "01-MI-AL-W 16.12.2021"                                                                           #(erst ab 16.12.21)
Alberichstraße = "24-MH-ALB 01.07.2015"
InvalidenstraßeOst = "03-MI-SAN-O 01.06.2015"
InvalidenstraßeWest = "03-MI-SAN-W 01.06.2015"
OberbaumbrückeWest = "05-FK-OBB-W 01.06.2015"                                                                       #(erst 02.05.2022)
OberbaumbrückeOst = "05-FK-OBB-O 01.06.2015"                                                                        #(erst 02.05.2022)
BreitenbachplatzOst = "17-SZ-BRE-O 01.05.2016"
BreitenbachplatzWest = "17-SZ-BRE-W 01.05.2016"
#avg_Klosterstraße, avg_Jannowitzbrücke, avg_Invalidenstraße, avg_Oberbaumbrücke, avg_Breitenbachplatz

#df = data[data["Zählstation"] == Alberichstraße]
df['Zeitpunkt'] = pd.to_datetime(df['Zeitpunkt'])


#print(station_df.describe())

#sns.histplot(data=station_df, x="Wert", bins=20, kde=False)  # bins: Anzahl der Säulen

#plt.xlabel("Wert")
#plt.ylabel("Anzahl")
#plt.title(f"Histogramm der Werte für die Station")

#plt.show()

#fig = model.plot_components(forecast_test)
#plt.show()








#PLOT tägliche Saisonalität aggregiert
# df['Stunde'] = df['Zeitpunkt'].dt.hour
# daily = df.groupby('Stunde')['Wert'].mean().reset_index()
# plt.figure(figsize=(10,5))
# sns.lineplot(data=daily, x='Stunde', y='Wert', marker='o')
# plt.ylabel("Durchschnittlicher Wert")
# plt.xlabel("Stunde")
# plt.xticks(range(0,24))
# plt.grid(True)
# plt.show()

#PLOT wöchentliche Saisonalität aggregiert
# df['Wochentag'] = df['Zeitpunkt'].dt.dayofweek  # Montag=0
# weekly = df.groupby('Wochentag')['Wert'].mean().reset_index()
# weekdays = ['Montag','Dienstag','Mittwoch','Donnerstag','Freitag','Samstag','Sonntag']
# weekly['Wochentag_name'] = weekly['Wochentag'].map(lambda x: weekdays[x])
# plt.figure(figsize=(10,5))
# sns.lineplot(data=weekly, x='Wochentag_name', y='Wert', marker='o')
# plt.xlabel("Wochentag")
# plt.ylabel("Durchschnittlicher Wert")
# plt.grid(True)
# plt.show()

#PLOT Jährliche Saisonalität
df['Monat'] = df['Zeitpunkt'].dt.month
monthly_agg = df.groupby('Monat')['Wert'].mean().reset_index()
plt.figure(figsize=(12,5))
sns.lineplot(data=monthly_agg, x='Monat', y='Wert', marker='o')
plt.xlabel("Monat")
plt.ylabel("Durchschnittlicher Wert")
plt.xticks(range(1,13))  # Monate 1-12
plt.grid(True)
plt.show()



