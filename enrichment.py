import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("finalData.csv", parse_dates=["Zeitpunkt"])

data["Wochentag"] = data["Zeitpunkt"].dt.day_name()                                                                 #Wochentage dazu, funktioniert

data['mergeDatum'] = data['Zeitpunkt'].dt.date
data['mergeDatum'] = pd.to_datetime(data['mergeDatum'])

feiertage = pd.read_csv("Feiertage DE.csv", sep=';', parse_dates=["Datum"])                                         #csv Feiertage
berliner_feiertage = feiertage[feiertage["Abkuerzung"].isin(["BE"])]                                                #nur Berliner Feiertage
data = data.merge(berliner_feiertage[['Datum', "Feiertag"]], left_on="mergeDatum", right_on="Datum", how="left")     #verbinden
data = data.drop(columns=['Datum'])                                                                                 #Datum löschen
data = data.drop(columns=['mergeDatum'])
data["Arbeitstag"] = np.where((data['Wochentag'] != "Sunday") & (data['Feiertag'].isna()), 1, 0)                    #Arbeitstag 0/1 dazu
data["Wochenende"] = np.where((data['Wochentag'] == "Sunday") | (data['Wochentag'] == "Saturday"), 1, 0)                    #WE 0/1 dazu


arenaAndHolidays = pd.read_csv("arenaAndHolidays.csv", parse_dates=["Datum"])
data = data.merge(arenaAndHolidays[['Datum', "UA Konzert", "UA Basketball", "UA Darts", "UA Eishockey",
                                    "UA Sonstige", "Ferien"]], left_on="Zeitpunkt", right_on="Datum", how="left")
data = data.drop(columns=['Datum'])
data = data.drop(columns=['Ferien'])                                                                                #unvollständige Ferien weg
data["Event"] = np.where((data['UA Konzert'].isna()) & (data['UA Basketball'].isna()) & (data['UA Darts'].isna())
                         & (data["UA Eishockey"].isna()) & (data['UA Sonstige'].isna()), 0, 1)                      #Event 0/1
data.loc[:, ["UA Konzert", "UA Basketball", 'UA Darts', "UA Eishockey", 'UA Sonstige']] = data.loc[:, ["UA Konzert", "UA Basketball", 'UA Darts', "UA Eishockey", 'UA Sonstige']].fillna(0)


holidays = pd.read_csv("holidays.csv", parse_dates=["Datum"])                                                       #vollständiger Ferien hinzu
holidays["Datum"] = pd.to_datetime(holidays["Datum"], errors="coerce")
data = data.merge(holidays[['Datum', "Ferien"]], left_on="Zeitpunkt", right_on="Datum", how="left")
data = data.drop(columns=['Datum'])

niederschlag = pd.read_csv("Niederschlag.csv", sep=';', parse_dates=["MESS_DATUM"], date_format="%Y%m%d%H")         #kein Datum type anpassung nötig
data = data.merge(niederschlag[['MESS_DATUM', "  R1", "RS_IND"]], left_on="Zeitpunkt", right_on="MESS_DATUM", how="left")
data = data.drop(columns=['MESS_DATUM'])
data = data.rename(columns={"  R1": "Niederschlag in mm"})
data = data.rename(columns={"RS_IND": "Niederschlag?"})

tempFeucht = pd.read_csv("temp feucht.csv", sep=';', parse_dates=["MESS_DATUM"], date_format="%Y%m%d%H")         #kein Datum type anpassung nötig
data = data.merge(tempFeucht[['MESS_DATUM', "TT_TU", "RF_TU"]], left_on="Zeitpunkt", right_on="MESS_DATUM", how="left")
data = data.drop(columns=['MESS_DATUM'])
data = data.rename(columns={"TT_TU": "Temperatur"})
data = data.rename(columns={"RF_TU": "Luftfeuchtigkeit"})

wind = pd.read_csv("wind.csv", sep=';', parse_dates=["MESS_DATUM"], date_format="%Y%m%d%H")         #kein Datum type anpassung nötig
data = data.merge(wind[['MESS_DATUM', "   F"]], left_on="Zeitpunkt", right_on="MESS_DATUM", how="left")
data = data.drop(columns=['MESS_DATUM'])
data = data.rename(columns={"   F": "Windgeschwindigkeit in m/s"})


data.loc[(data["Zählstation"] == "03-MI-SAN-O 01.06.2015") & (data["Zeitpunkt"].between("2018-04-20 10:00:00", "2018-04-20 13:00:00")),"Wert"] = np.nan            #Qualitätsmängel entfernen
data.loc[(data["Zählstation"] == "03-MI-SAN-W 01.06.2015") & (data["Zeitpunkt"].between("2018-04-20 10:00:00", "2018-04-20 13:00:00")),"Wert"] = np.nan
data.loc[(data["Zählstation"] == "03-MI-SAN-O 01.06.2015") & (data["Zeitpunkt"].between("2018-10-08 09:00:00", "2019-10-30 11:00:00")),"Wert"] = np.nan
data.loc[(data["Zählstation"] == "02-MI-JAN-N 01.04.2015") & (data["Zeitpunkt"].between("2019-10-09 14:00:00", "2019-10-10 07:00:00")),"Wert"] = np.nan
data.loc[(data["Zählstation"] == "02-MI-JAN-S 01.04.2015") & (data["Zeitpunkt"].between("2019-10-09 14:00:00", "2019-10-10 07:00:00")),"Wert"] = np.nan
data.loc[(data["Zählstation"] == "07-FK-ST 22.04.2024") & (data["Zeitpunkt"].between("2024-04-22 00:00:00", "2024-04-22 00:00:00")),"Wert"] = np.nan
data.loc[(data["Zählstation"] == "02-MI-JAN-S 01.04.2015") & (data["Zeitpunkt"].between("2024-06-26 00:00:00", "2024-12-31 23:00:00")),"Wert"] = np.nan
data.loc[(data["Zählstation"] == "17-SZ-BRE-O 01.05.2016") & (data["Zeitpunkt"].between("2021-07-02 16:00:00", "2021-07-09 09:00:00")),"Wert"] = np.nan


data["Stunde"] = data["Zeitpunkt"].dt.hour
data["Tag_im_Jahr"] = data["Zeitpunkt"].dt.dayofyear
data["Monat"] = data["Zeitpunkt"].dt.month
data["Jahr"] = data["Zeitpunkt"].dt.year
data["Zeitpunkt_num"] = data["Zeitpunkt"].astype("int64") // 10 ** 9

data.to_csv("WORKDATAold.csv", index=False)




