import pandas as pd

data = pd.read_csv("WORKDATA.csv", parse_dates=["Zeitpunkt"], low_memory=False)

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

stations = [KlosterstraßeSüd, KlosterstraßeNord, JannowitzbrückeNord, JannowitzbrückeSüd, KarlMarxAllee, Alberichstraße,
            InvalidenstraßeOst, InvalidenstraßeWest, OberbaumbrückeWest, OberbaumbrückeOst, BreitenbachplatzOst, BreitenbachplatzWest,
            "avg_Klosterstraße", "avg_Jannowitzbrücke", "avg_Invalidenstraße", "avg_Oberbaumbrücke", "avg_Breitenbachplatz"]

station_dates = {KlosterstraßeSüd: "2017-01-01",
            KlosterstraßeNord: "2017-01-01",
            JannowitzbrückeNord: "2017-01-01",
            JannowitzbrückeSüd: "2017-01-01",
            KarlMarxAllee: "2021-12-16",
            Alberichstraße: "2017-01-01",
            InvalidenstraßeOst: "2020-01-01",
            InvalidenstraßeWest: "2020-01-01",
            OberbaumbrückeWest: "2022-05-02",
            OberbaumbrückeOst: "2022-05-02",
            BreitenbachplatzOst: "2017-01-01",
            BreitenbachplatzWest: "2017-01-01",
            "avg_Klosterstraße": "2017-01-01",
            "avg_Jannowitzbrücke": "2017-01-01",
            "avg_Invalidenstraße": "2020-01-01",
            "avg_Oberbaumbrücke": "2022-05-02",
            "avg_Breitenbachplatz": "2017-01-01"}

for station in stations:
    print("\n====================================================")
    print(f"Station: {station}")

    df = data[data["Zählstation"] == station].copy()
    start_date = pd.Timestamp(station_dates[station])
    df = df[df["Zeitpunkt"] >= start_date]

    df = df.dropna(subset=["Wert"])

    print(df["Wert"].quantile(0.75)-df["Wert"].quantile(0.25))