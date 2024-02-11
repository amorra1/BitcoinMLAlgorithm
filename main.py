# region imports
from AlgorithmImports import *
from tensorflow.keras.models import Sequential
import json


# endregion

class MuscularGreenCoyote(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)  # set start date
        self.SetEndDate(2020, 1, 1)  # set end date

        # Get model (copy from ipynb)
        model_key = 'bitcoin_price_predictor'
        if self.ObjectStore.ContainsKey(model_key):
            model_str = self.ObjectStore.Read(model_key)
            config = json.loads(model_str)['config']
            self.model = Sequential.from_config(config)

        # setup stading platform
        self.SetBrokerageModel(BrokerageName.Bitfinex, AccountType.Margin)  # Crypto brokerage
        self.SetCash(100000)  # set strategy cash
        self.symbol = self.AddCrypto("BTCUSD", Resolution.Daily).Symbol
        self.SetBenchmark(self.symbol)  # allows to compare strategy

    # Strategy
    def OnData(self, data):
        if self.GetPrediction() == "Up":
            self.SetHoldings(self.symbol, 1)
        else:
            self.SetHoldings(self.symbol, 0)

    def GetPrediction(self):
        # instead of history requests, use rolling window for more efficiency
        df = self.History(self.symbol, 40).loc[self.symbol]
        df_change = df[["close", "open", "high", "low", "volume"]].pct_change().dropna()
        model_input = []
        # turn history into right input formal for model
        for index, row in df_change.tail(30).iterrows():
            model_input.append(np.array(row))
        model_input = np.array([model_input])
        if round(self.model.predict(model_input)[0][0]) == 1:
            return "Up"
        else:
            return "Down"