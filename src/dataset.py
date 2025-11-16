# Code refined from the reproduction of the GPT4TS paper done by liaoyuhua for FirstRate dataset formatting
# Reference: https://github.com/liaoyuhua/GPT-TS
# Credit to the authors of the paper and repository.

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import StandardScaler

TOKEN_MAPPINGS = {
    "event": {"NONE": 0, "MEDIUM": 1, "HIGH": 2},
    "session": {"None": 0, "NY": 1, "London": 2, "Both": 3, "London New": 4}
}


def calculate_rsi(prices, period=14):
  """
  Calculates the Relative Strength Index (RSI) for a given list of prices.
 
  Arguments:
      prices (list): list of closing prices/returns/log returns
      period (int) : number of periods for the moving average (default: 14)
 
  Returns:
      pd.Series: contains the RSI value
  """
 
  delta = prices.diff()
  delta = delta.dropna()  # Remove NaN from the first difference
  up, down = delta.clip(lower=0), delta.clip(upper=0, lower=None)  # Separate gains and losses
 
  ema_up = up.ewm(alpha=1/period, min_periods=period).mean()  # Exponential Moving Average for gains
  ema_down = down.abs().ewm(alpha=1/period, min_periods=period).mean()  # EMA for absolute losses
 
  rs = ema_up / ema_down  # Average gain / Average loss
  rsi = 100 - 100 / (1 + rs)  # Calculate RSI
 
  return rsi

def session_label(row):
    """
    Classifies the market session associated with current timepoint.
    Arguments:
        row (pd.Series): info about current timepoint, containing local London and New York time
    Returns:
        str: the market session associated with current timepoint, including:
             "London Opening" (London-Tokyo overlap), "London Only", "London-NY overlap", "NY only", and "Tokyo"
    """
    london_open = 7 <= row['london_time'].hour < 16
    ny_open = 8 <= row['ny_time'].hour < 17
    london_new_open = (row['london_time'].hour == 7) or (row['london_time'].hour == 8 and row['london_time'].minute == 0)

    if london_open and ny_open:
        return 'Both'
    elif london_open and not ny_open:
        if london_new_open:
            return "London New"
        else:
            return 'London'
    elif ny_open and not london_open:
        return 'NY'
    else:
        return 'None'


class Dataset_Rates_30M(Dataset):
    """
    PyTorch dataset for currency exchange rate time-series.

    Provides input sequences and target values for forecasting tasks. Each sample contain the historical
    target values and possibly other predictors as inputs, while the output is a univariate time-series
    of future target values.
    """
    def __init__(
        self,
        root_path,
        rates,
        flag="train",
        size=None,
        features="S",
        target="vola",
        scale=True,
        inverse=False,
        cols=None,
        use_technical=True,
        use_events=True,
        use_interday=True,
        fine_tuning_pct=None,
        use_explainable=False
    ):
        """
        Arguments:
            root_path (str)           : data path of the project
            rates (str)               : the exchange rate considered (currently support gbp_usd, eur_usd, usd_chf and usd_jpy)
            flag (str)                : train/val/test representing the training, validation and testing period of the dataset
            size (tuple)              : 2-tuple containing the lookback and future horizons
            features (str)            : S/MS.
                                        - S: Univariate time-series input
                                        - MS: Multivariate time-series input
            target (str)              : target variable of the time-series
            scale (bool)              : if true, standardises time-series features using mean and std from training period
            use_technical (bool)      : whether to include technical indicators in the input time-series
            use_events (bool)         : whether to include macroeconomic event indicators in the input time-series
            use_interday (bool)       : whether to include interday volatility in the input time-series
            fine_tunning_pct (float)  : percentage of training data used (default: 100% of training data)
            use_explainable (bool)    : whether to include semantic tokens in input data.
        """
        if size is None:
            self.seq_len = 96
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.use_explainable = use_explainable
        self.rates = rates

        self.root_path = root_path
        self.data_path = f"{self.rates[0].lower()}_{self.rates[1].lower()}_30m_20_24_ext_v1.csv"

        self.use_technical = use_technical
        self.use_events = use_events
        self.use_interday = use_interday
        self.fine_tuning_pct = fine_tuning_pct
        if flag != "train" and self.fine_tuning_pct is not None:
            print("Warning: fine tuning percentage specified for non-training dataset")
        
        self.__read_data__()

    def __read_data__(self):
        """
        Extracts data from FirstRate's exchange rate dataset format.
        """
        
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        df_data = []
        raw_df_data = []
        df_week_groups = df_raw.groupby("week_num")

        for week_num, week_data in df_week_groups:

            if len(week_data) < 230:
                # more than 10 missing canddlestick bars in this week, ignore
                continue
            
            week_data["log_return"] = np.log(week_data["Close"] / week_data["Close"].shift(1))
            week_data["return"] = week_data["Close"] / week_data["Close"].shift(1)
            week_data["vola"] = np.log(week_data["High"] / week_data["Low"])


            # technical indicators
            self.TECH_INDICATORS = ["middle_band", "upper_band", "lower_band", "momentum", "acceleration", "ema", "rsi"]
            # Bollinger bands
            T = 12
            D = 2
            week_data["middle_band"] = week_data["log_return"].rolling(window=T).mean()
            rolling_std = week_data['log_return'].rolling(window=T).std()
            week_data['upper_band'] = week_data['middle_band'] + D * rolling_std
            week_data['lower_band'] = week_data['middle_band'] - D * rolling_std

            # momentum and acceleration
            momentum_lookback = 12
            week_data['momentum'] = week_data['log_return'] - week_data['log_return'].shift(momentum_lookback)
            week_data['acceleration'] = week_data['momentum'] - week_data['momentum'].shift(12)

            # EMA, RSI
            RSI_n = 14
            week_data['ema'] = week_data['log_return'].ewm(span=T, adjust=False).mean()
            week_data["rsi"] = calculate_rsi(week_data["log_return"], RSI_n)


            # interday volatility
            self.INTERDAY_VOLAS = [f"prev_vola-{i}" for i in range(1, 6)]

            # economic events
            self.ECONOMIC_EVENTS = [f"{impact}-{self.rates[1].upper()}" for impact in ["LOW", "MEDIUM", "HIGH"]] + [f"{impact}-{self.rates[0].upper()}" for impact in ["LOW", "MEDIUM", "HIGH"]]
            

            # drops early days
            week_data = week_data.dropna()

            self.future_feature_index = 0
            if self.features in ["M"]:
                raise NotImplementedError()
            elif self.features == "S":
                raw_week_data = week_data.copy()
                week_data = week_data[[self.target]]
            elif self.features == "MS":

                used_features = []
                if self.use_technical:
                    used_features += self.TECH_INDICATORS + ["log_return"] + ["Volume"]
                    self.future_feature_index += len(self.TECH_INDICATORS + ["log_return"] + ["Volume"])
                if self.use_events:
                    used_features += self.ECONOMIC_EVENTS
                if self.use_interday:
                    used_features += self.INTERDAY_VOLAS

                used_features.append(self.target)

                raw_week_data = week_data.copy().reset_index(drop=True)
                week_data = week_data[used_features]

            df_data.append(week_data)
            raw_df_data.append(raw_week_data)

        if self.use_explainable:
            for i in range(len(raw_df_data)):
                cur_df = raw_df_data[i]
                cur_df['london_time'] = pd.to_datetime(cur_df['Gmt time']).dt.tz_convert('Europe/London')
                cur_df['ny_time'] = pd.to_datetime(cur_df['Gmt time']).dt.tz_convert('America/New_York')
                cur_df["session"] = cur_df.apply(session_label, axis=1)

         
        # train - val - test split
        train_start_week = 0
        if self.fine_tuning_pct is not None and self.fine_tuning_pct < 1:  
          train_start_week = int(0.7 * (1 - self.fine_tuning_pct - 0.001) * len(df_data))
      
        val_start_week = int(0.7 * len(df_data))
        test_start_week = int(0.85 * len(df_data))
        borders = [train_start_week, val_start_week, test_start_week, len(df_data)]
        
        if self.scale:
            train_target = pd.concat([df_data[i] for i in range(train_start_week, val_start_week)]).values
            self.train_target = train_target
            self.scaler.fit(train_target)
        self.data = []
        if self.use_explainable:
            self.raw_data = []
    
        for i in range(borders[self.set_type], borders[self.set_type+1]):
            week_data = df_data[i].values
            if self.scale:
                week_data = self.scaler.transform(week_data)
            self.data.append(week_data)
            if self.use_explainable:
                self.raw_data.append(raw_df_data[i])
        self.week_lens = [len(week_data) for week_data in self.data]
        self.data_len = [(week_len - self.seq_len - self.pred_len + 1) for week_len in self.week_lens]
        self.cumsum = np.cumsum(self.data_len)
        self.tot_len = sum(self.data_len)

    
    def __len__(self):
        return self.tot_len

    def _find_week_index(self, index):
        """
        Finds the week corresponding with the timepoint with the given `index` in the data.
        """
        l = 0; r = len(self.cumsum)
        while l <= r:
            mid = (l + r) // 2
            if index >= self.cumsum[mid]:
                l = mid + 1
            else:
                r = mid - 1
        return l

    def __getitem__(self, index):
        """
        Input shape:
        - features=S : (1, L_in)
        - features=MS: (N, L_in)

        Output shape: (1, L_out)
        """
        week_index = self._find_week_index(index)

        if week_index > 0:
            day_index = index - self.cumsum[week_index-1]
        else:
            day_index = index
        
        s_begin = day_index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[week_index][s_begin:s_end]
        seq_y = self.data[week_index][r_begin:r_end][:, -1:]


        if self.use_explainable:
            
            next_tp = self.raw_data[week_index].iloc[r_begin]
            session = torch.tensor(
                TOKEN_MAPPINGS["session"][next_tp["session"]], dtype=torch.long
            )
            event_token = None
            if next_tp[f"HIGH-{self.rates[1].upper()}"] or next_tp[f"HIGH-{self.rates[0].upper()}"]:
                event_token = TOKEN_MAPPINGS["event"]["HIGH"]
            elif next_tp[f"MEDIUM-{self.rates[1].upper()}"] or next_tp[f"MEDIUM-{self.rates[0].upper()}"]:
                event_token = TOKEN_MAPPINGS["event"]["MEDIUM"]
            else:
                event_token = TOKEN_MAPPINGS["event"]["NONE"]
            event = torch.tensor(event_token, dtype=torch.long)
            
            
        
        x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)
        y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)
        
        
        if self.use_explainable:
            return (x, {"market_session": session, "event": event}), y
        else:
            return x, y

    def inverse_transform(self, data):
        """
        Re-scales time-series back to their initial values.
        """
        return self.scaler.inverse_transform(data)