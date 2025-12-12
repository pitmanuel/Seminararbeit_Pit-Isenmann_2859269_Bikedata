

import pandas as pd

import requests
from io import StringIO

bikedata = pd.read_excel("gesamtdatei-stundenwerte - erste Bearbeitung.xlsx", sheet_name=None)      #funktioniert

all_data = pd.concat(bikedata.values(), ignore_index=True)

#all_data.to_excel("alle_sheets_zusammen.xlsx", index=False)

