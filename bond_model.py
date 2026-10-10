import numpy as np
import pandas as pd

class BondYieldModel:
    def __init__(self, china_yields, us_yields):
        self.china = pd.DataFrame({'yield': china_yields})
        self.us = pd.DataFrame({'yield': us_yields})
        
        self.thresholds = {'china': 1.2229089, 'us': 2.6969138}
        
        self._prepare_data(self.china, 'china')
        self._prepare_data(self.us, 'us')

    def _prepare_data(self, df, country):
        df['change_bp'] = df['yield'].diff() * 100
        df['days_diff'] = df.index.to_series().diff().dt.days
        
        df = df[df['days_diff'] <= 4].copy()
        df.dropna(subset=['change_bp'], inplace=True)
        
        threshold = self.thresholds[country]
        
        conditions = [
            df['change_bp'] >= threshold,
            df['change_bp'] <= -threshold
        ]
        choices = ['Up', 'Down']
        df['state'] = np.select(conditions, choices, default='Flat')
        
        if country == 'china':
            self.china = df
        else:
            self.us = df

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
        
        rolling_rms = df['change_bp'].shift(1).rolling(window=window).apply(
            lambda x: np.sqrt(np.mean(x**2)), raw=True
        )
        return rolling_rms
