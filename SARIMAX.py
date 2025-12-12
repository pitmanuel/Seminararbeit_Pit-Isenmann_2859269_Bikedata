import optuna
import pandas as pd
import pip
import statsmodels as statsmodels
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

#FUNKTIONIERT ABER EINE STATION 15min

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


df = data[data["Zählstation"] == Alberichstraße].copy()


df = df.set_index("Zeitpunkt").sort_index()
df = df.asfreq("h")   # falls stündliche Daten
df["Wert"] = df["Wert"].fillna(df["Wert"].median())

y = df["Wert"]

numerical_cols = [
    "Arbeitstag", "Wochenende", "Event", "Niederschlag in mm", "Niederschlag?",
    "Temperatur", "Luftfeuchtigkeit",
    "Windgeschwindigkeit in m/s", "Zeitpunkt_num", "Stunde", "Tag_im_Jahr", "Monat", "Jahr", "lag_1h", "lag_24h", "lag_1week"]

imputer = SimpleImputer(strategy="median")
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

categorical_cols = ["Wochentag", "Feiertag", "Ferien"]          #wenn diese raus, deutlich schneller (vlt so ne minute nur noch)
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
#exog = pd.concat([df[numerical_cols], fs], axis=1)

bool_cols = exog.select_dtypes(include='bool').columns
exog[bool_cols] = exog[bool_cols].astype(int)

split_date = "2023-01-01"
y_train = df.loc[df.index < split_date, "Wert"]
y_test  = df.loc[df.index >= split_date, "Wert"]
exog_train = exog.loc[df.index < split_date]
exog_test  = exog.loc[df.index >= split_date]


#optuna nach 20 min ohne fertigen trail abgebrochen
# def objective(trial):
#     p = trial.suggest_int("p", 0, 3)
#     d = trial.suggest_int("d", 0, 2)
#     q = trial.suggest_int("q", 0, 3)
#     P = trial.suggest_int("P", 0, 2)
#     D = trial.suggest_int("D", 0, 1)
#     Q = trial.suggest_int("Q", 0, 2)
#     s = 168  # wöchentliche Saisonalität (24*7)
#
#     try:
#         model = SARIMAX(
#             y_train,
#             exog=exog_train,
#             order=(p, d, q),
#             seasonal_order=(P, D, Q, s),
#             enforce_stationarity=False,
#             enforce_invertibility=False
#         )
#         results = model.fit(disp=False)
#         y_pred = results.get_forecast(steps=len(y_test), exog=exog_test).predicted_mean
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         return rmse
#     except:
#         return np.inf
#
#
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)  # Anzahl der Versuche kann angepasst werden
#
# print("Beste Hyperparameter:", study.best_params)
#
# # --- Modell mit besten Parametern ---
# best_params = study.best_params
# best_model = SARIMAX(
#     y_train,
#     exog=exog_train,
#     order=(best_params["p"], best_params["d"], best_params["q"]),
#     seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], 168),
#     enforce_stationarity=False,
#     enforce_invertibility=False
# )
# results = best_model.fit(disp=False)
#
# y_pred = results.get_forecast(steps=len(y_test), exog=exog_test).predicted_mean
#
# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# cv = np.std(y_test - y_pred) / np.mean(y_test)
#
# print("R²: ", r2)
# print("RMSE:", rmse)
# print("MAE:", mae)
# print("CV (RMSE/mean):", cv)


model = SARIMAX(y_train,
                exog=exog_train,
                order=(2,0,2),
                seasonal_order=(0,0,0,0),
                enforce_stationarity=False,
                enforce_invertibility=False)

#res = model.fit()

#model = SARIMAX(
#    y,
#    exog=exog,
#    order=(1,0,1),        # ARIMA
#    seasonal_order=(1,1,1,24*7),  # wöchentliche Saisonalität
#    enforce_stationarity=False,
#    enforce_invertibility=False
#)


results = model.fit()
#print(results.summary())

y_pred = results.get_forecast(steps=len(y_test), exog=exog_test).predicted_mean

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
cv = np.std(y_test - y_pred) / np.mean(y_test)

print("R²: ", r2)
print("RMSE:", rmse)
print("MAE:", mae)
print("CV (RMSE/mean):", cv)




