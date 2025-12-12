import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

data = pd.read_csv("WORKDATAold.csv", parse_dates=["Zeitpunkt"], low_memory=False)

KlosterstraßeSüd = "15-SP-KLO-S 01.06.2016"
KlosterstraßeNord = "15-SP-KLO-N 01.06.2016"
JannowitzbrückeNord = "02-MI-JAN-N 01.04.2015"
JannowitzbrückeSüd = "02-MI-JAN-S 01.04.2015"
InvalidenstraßeOst = "03-MI-SAN-O 01.06.2015"
InvalidenstraßeWest = "03-MI-SAN-W 01.06.2015"
OberbaumbrückeWest = "05-FK-OBB-W 01.06.2015"                                                                       #(erst 02.05.2022)
OberbaumbrückeOst = "05-FK-OBB-O 01.06.2015"
BreitenbachplatzOst = "17-SZ-BRE-O 01.05.2016"
BreitenbachplatzWest = "17-SZ-BRE-W 01.05.2016"

df_subset = data[data["Zählstation"].isin([KlosterstraßeSüd, KlosterstraßeNord])].copy()            #Erstellung Station avg_Klosterstraße
avg_values = df_subset.groupby("Zeitpunkt")["Wert"].mean()
df_new = df_subset[df_subset["Zählstation"] == KlosterstraßeSüd].copy()
df_new["Wert"] = avg_values.values
df_new["Zählstation"] = "avg_Klosterstraße"
data = pd.concat([data, df_new], ignore_index=True)

df_subset = data[data["Zählstation"].isin([JannowitzbrückeNord, JannowitzbrückeSüd])].copy()            #Erstellung Station avg_Jannowitzbrücke
avg_values = df_subset.groupby("Zeitpunkt")["Wert"].mean()
df_new = df_subset[df_subset["Zählstation"] == JannowitzbrückeNord].copy()
df_new["Wert"] = avg_values.values
df_new["Zählstation"] = "avg_Jannowitzbrücke"
data = pd.concat([data, df_new], ignore_index=True)

df_subset = data[data["Zählstation"].isin([InvalidenstraßeOst, InvalidenstraßeWest])].copy()            #Erstellung Station avg_Invalidenstraße
avg_values = df_subset.groupby("Zeitpunkt")["Wert"].mean()
df_new = df_subset[df_subset["Zählstation"] == InvalidenstraßeOst].copy()
df_new["Wert"] = avg_values.values
df_new["Zählstation"] = "avg_Invalidenstraße"
data = pd.concat([data, df_new], ignore_index=True)

df_subset = data[data["Zählstation"].isin([OberbaumbrückeWest, OberbaumbrückeOst])].copy()            #Erstellung Station avg_Oberbaumbrücke
avg_values = df_subset.groupby("Zeitpunkt")["Wert"].mean()
df_new = df_subset[df_subset["Zählstation"] == OberbaumbrückeWest].copy()
df_new["Wert"] = avg_values.values
df_new["Zählstation"] = "avg_Oberbaumbrücke"
data = pd.concat([data, df_new], ignore_index=True)

df_subset = data[data["Zählstation"].isin([BreitenbachplatzOst, BreitenbachplatzWest])].copy()            #Erstellung Station avg_Breitenbachplatz
avg_values = df_subset.groupby("Zeitpunkt")["Wert"].mean()
df_new = df_subset[df_subset["Zählstation"] == BreitenbachplatzOst].copy()
df_new["Wert"] = avg_values.values
df_new["Zählstation"] = "avg_Breitenbachplatz"
data = pd.concat([data, df_new], ignore_index=True)

data["lag_1h"] = data.groupby("Zählstation")["Wert"].shift(1)
data["lag_24h"] = data.groupby("Zählstation")["Wert"].shift(24)
data["lag_1week"] = data.groupby("Zählstation")["Wert"].shift(24*7)


data.to_csv("WORKDATA.csv", index=False)
