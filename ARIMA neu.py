import optuna
import pandas as pd
from optuna.samplers import GridSampler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


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

numerical_cols = [
    "Arbeitstag", "Wochenende", "Event", "Niederschlag in mm", "Niederschlag?",
    "Temperatur", "Luftfeuchtigkeit",
    "Windgeschwindigkeit in m/s", "Zeitpunkt_num", "Stunde", "Tag_im_Jahr", "Monat", "Jahr", "lag_1h", "lag_24h", "lag_1week"]

categorical_cols = ["Wochentag", "Feiertag", "Ferien"]

for station in stations:
    print("\n====================================================")
    print(f"Station: {station}")

    df = data[data["Zählstation"] == station].copy()

    start_date = pd.Timestamp(station_dates[station])
    df = df[df["Zeitpunkt"] >= start_date]

    df = df.set_index("Zeitpunkt").sort_index()
    df = df.asfreq("h")
    df["Wert"] = df["Wert"].fillna(df["Wert"].median())

    y = df["Wert"]


    imputer = SimpleImputer(strategy="median")
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    df[categorical_cols] = df[categorical_cols].fillna("Missing")                                               #NaN durch "Missing" ersetzt
    df_dummies = pd.get_dummies(df[categorical_cols], drop_first=True)                                          #verwandelt in numerische Spalten

    def fourier_series(index, period, K):
        t = np.arange(len(index))
        df_fs = {}
        for k in range(1, K+1):
            df_fs[f"sin_{period}_{k}"] = np.sin(2*np.pi*k*t/period)
            df_fs[f"cos_{period}_{k}"] = np.cos(2*np.pi*k*t/period)
        return pd.DataFrame(df_fs, index=index)

    fs = fourier_series(df.index, period=168, K=6)

    exog = pd.concat([df[numerical_cols], df_dummies, fs], axis=1)

    bool_cols = exog.select_dtypes(include='bool').columns
    exog[bool_cols] = exog[bool_cols].astype(int)

    split_date = "2023-01-01"
    y_train = df.loc[df.index < split_date, "Wert"]
    y_test = df.loc[df.index >= split_date, "Wert"]
    exog_train = exog.loc[df.index < split_date]
    exog_test = exog.loc[df.index >= split_date]

    param_grid = {
        "p": list(range(4)),  # 0,1,2,3
        "q": list(range(4))   # 0,1,2,3
    }
    sampler = GridSampler(param_grid)

    def objective(trial):
        p = trial.suggest_categorical("p", param_grid["p"])
        d = 0
        q = trial.suggest_categorical("q", param_grid["q"])

        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)
            y_pred = results.get_forecast(steps=len(y_test), exog=exog_test).predicted_mean
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            return rmse
        except:
            return np.inf

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=len(param_grid["p"]) * len(param_grid["q"]))

    print("Beste Hyperparameter:", study.best_params)

    best_params = study.best_params
    best_model = SARIMAX(
        y_train,
        exog=exog_train,
        order=(best_params["p"], 0, best_params["q"]),
        seasonal_order=(0, 0, 0, 0),                    #saisonal raus
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = best_model.fit(disp=False)

    y_pred = results.get_forecast(steps=len(y_test), exog=exog_test).predicted_mean

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    cv = np.std(y_test - y_pred) / np.mean(y_test)

    print("R²: ", r2)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("CV (RMSE/mean):", cv)

