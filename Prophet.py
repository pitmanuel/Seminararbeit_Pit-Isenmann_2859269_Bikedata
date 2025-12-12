import optuna
import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#sind raus: Stunde, Monat, Jahr, TagimJahr, wochentag, feiertag, ferien, Zeitpunkt_num

data = pd.read_csv("WORKDATA.csv", parse_dates=["Zeitpunkt"], low_memory=False)

data = data.drop(columns=['Stunde', 'Monat', 'Jahr', 'Tag_im_Jahr', 'Wochentag'])


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
    df = df.drop(columns=['Zählstation'])

    df.rename(columns={"Zeitpunkt": "ds", "Wert": "y"}, inplace=True)
    df = df.dropna(subset=['y'])

    feiertage_df = df[['ds', 'Feiertag']].copy()
    feiertage_df = feiertage_df[feiertage_df['Feiertag'].notna()]
    feiertage_df = feiertage_df[feiertage_df['Feiertag'] != "Missing"]
    feiertage_df.rename(columns={'Feiertag': 'holiday'}, inplace=True)
    feiertage_df = feiertage_df.drop_duplicates()

    ferien_df = df[['ds', 'Ferien']].copy()
    ferien_df = ferien_df[ferien_df['Ferien'].notna()]
    ferien_df = ferien_df[ferien_df['Ferien'] != "Missing"]
    ferien_df.rename(columns={'Ferien': 'holiday'}, inplace=True)
    ferien_df = ferien_df.drop_duplicates()

    holidays_df = pd.concat([feiertage_df, ferien_df])
    holidays_df = holidays_df.sort_values('ds').reset_index(drop=True)

    df = df.drop(columns=['Feiertag', 'Ferien'])

    num_features = [
        "Arbeitstag", "Wochenende", "UA Konzert", "UA Basketball", "UA Darts", "UA Eishockey", "UA Sonstige", "Event", "Niederschlag in mm", "Niederschlag?",
        "Temperatur", "Luftfeuchtigkeit",
        "Windgeschwindigkeit in m/s", "lag_1h", "lag_24h", "lag_1week"
    ]

    event_cols = ["UA Konzert", "UA Basketball", "UA Darts", "UA Eishockey", "UA Sonstige", "Event", "Arbeitstag", "Wochenende", "Niederschlag?"]
    for col in event_cols:
        df[col] = df[col].fillna(0)

    for col in num_features:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    regressors = [col for col in df.columns if col not in ['ds', 'y']]

    train_df = df[df['ds'] < '2023-01-01'].copy()
    test_df = df[df['ds'] >= '2023-01-01'].copy()
    train_df = train_df.sort_values("ds")
    test_df = test_df.sort_values("ds")

    def objective(trial):
        changepoint_prior_scale = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
        seasonality_prior_scale = trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True)
        holidays_prior_scale = trial.suggest_float("holidays_prior_scale", 0.01, 10.0, log=True)

        model = Prophet(
            holidays=holidays_df,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale
        )

        for col in regressors:
            model.add_regressor(col)

        model.fit(train_df)
        forecast = model.predict(test_df)

        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Beste Hyperparameter:", study.best_params)
    print("Bester RMSE:", study.best_value)

    best_params = study.best_params
    model = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        **best_params
    )
    for col in regressors:
        model.add_regressor(col)
    model.fit(train_df)

    forecast_test = model.predict(test_df)

    y_true = test_df['y'].values
    y_pred = forecast_test['yhat'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print("CV Test:", np.sqrt(mean_squared_error(y_true, y_pred)) / y_true.mean())
