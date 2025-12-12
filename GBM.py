import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

    df = df.dropna(subset=["Wert"])                                                                            #Zeilen bei Wert NaN raus

    df[categorical_cols] = df[categorical_cols].fillna("Missing")                                               #NaN durch "Missing" ersetzt
    df_dummies = pd.get_dummies(df[categorical_cols], drop_first=True)                                          #verwandelt in numerische Spalten

    X = pd.concat([df[numerical_cols], df_dummies], axis=1)
    y = df["Wert"]

    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])                                   #NaN von numerischen mit Median ersetzen

    train_mask = df["Zeitpunkt"] >= pd.Timestamp("2023-01-01")
    test_mask = df["Zeitpunkt"] < pd.Timestamp("2023-01-01")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
        }

        model = GradientBoostingRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Beste Hyperparameter:")
    print(study.best_params)

    model = GradientBoostingRegressor(**study.best_params, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    print("R² auf Testdaten:", r2_score(y_test, y_pred_test))
    print("RMSE Test:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print("MAE Test:", mean_absolute_error(y_test, y_pred_test))
    print("CV Test:", np.sqrt(mean_squared_error(y_test, y_pred_test)) / y_test.mean())
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False).head(30))
