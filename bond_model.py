import numpy as np
import pandas as pd

class BondYieldModel:
    def __init__(self, china_yields, us_yields):
        self.china = pd.DataFrame({'yield': china_yields})
        self.us = pd.DataFrame({'yield': us_yields})
        self._prepare_data()

    def _prepare_data(self):
        for df in [self.china, self.us]:
            df['change_bp'] = df['yield'].diff() * 100
            df.dropna(inplace=True)
            
            sd = df['change_bp'].std()
            threshold = 0.5 * sd
            
            conditions = [
                df['change_bp'] >= threshold,
                df['change_bp'] <= -threshold
            ]
            choices = ['Up', 'Down']
            df['state'] = np.select(conditions, choices, default='Flat')

    def get_markov_transition_matrix(self, country='china'):
        df = self.china if country == 'china' else self.us
        states = ['Down', 'Flat', 'Up']
        
        df['next_state'] = df['state'].shift(-1)
        transitions = pd.crosstab(df['state'], df['next_state'])
        
        for state in states:
            if state not in transitions.columns:
                transitions[state] = 0
            if state not in transitions.index:
                transitions.loc[state] = 0
                
        transitions = transitions.reindex(index=states, columns=states)
        
        smoothed_matrix = transitions.apply(lambda row: (row + 1) / (row.sum() + 3), axis=1)
        return smoothed_matrix

    def get_rolling_volatility(self, country='china', window=60):
        df = self.china if country == 'china' else self.us
        return df['change_bp'].shift(1).rolling(window=window).std()

if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start="2015-01-01", periods=2000, freq='B')
    
    mock_china = np.cumsum(np.random.normal(loc=0, scale=0.024, size=2000)) + 3.0
    mock_us = np.cumsum(np.random.normal(loc=0, scale=0.053, size=2000)) + 2.0

    model = BondYieldModel(
        china_yields=pd.Series(mock_china, index=dates),
        us_yields=pd.Series(mock_us, index=dates)
    )

    print(model.get_markov_transition_matrix('china').round(4))
    print(model.get_markov_transition_matrix('us').round(4))
    print(model.get_rolling_volatility('china', window=60).tail())
