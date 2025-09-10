import yfinance as yf
import numpy as np
import pandas as pd
import re
from datetime import datetime
from pandas.tseries.frequencies import to_offset
from itertools import product
import warnings
import math

###########
# Helpers #
###########

# a perplexity.ai function for converting offset strings
def normalize_period_string(period: str) -> str:
    """
    Convert custom frequency strings:
    - '1w' to '1W'
    - '3mo' to '3M'
    - '1y' to '1Y'
    - '2d' to '2D' (optional)
    - Also handles upper/lowercase
    """
    period = period.strip().lower()
    # Replace custom suffixes with pandas compatible ones
    if period.endswith("w"):
        return period[:-1] + "W"
    elif period.endswith("mo"):
        return period[:-2] + "M"
    elif period.endswith("y"):
        return period[:-1] + "Y"
    elif period.endswith("d"):
        return period[:-1] + "D"
    else:
        # fallback, return as-is for things like "5min"
        return period.upper()

# another perplexity.ai function for creating an offset
def date_minus_offset(date: str, offset: str) -> str:
    normalized = normalize_period_string(offset)
    match = re.match(r"(\d+)([MWYD])", normalized)
    if not match:
        raise ValueError(f"Invalid offset string: {offset}")
    num, unit = int(match.group(1)), match.group(2)
    # Map to DateOffset keyword
    kwargs = {}
    if unit == "Y":
        kwargs["years"] = num
    elif unit == "M":
        kwargs["months"] = num
    elif unit == "W":
        kwargs["weeks"] = num
    elif unit == "D":
        kwargs["days"] = num
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    new_date = pd.to_datetime(date) - pd.DateOffset(**kwargs)
    return str(new_date.date())

# a function for importing all tickers from Euronext spreadsheet
def get_euronext(file: str, end: str) -> list:
    data = pd.read_excel(file)
    if 'Symbol' not in data.columns:
        raise ValueError(f"Couldn't find a \'Symbol\' column in {file}.")
    data = data.loc[~data.Symbol.isna()]
    data['Symbol'] = data['Symbol'].astype(str) + end
    return data['Symbol'].to_list()

# a function for osebx shipping as of 29.08.25
def get_osebx_shipping():
    stocks = "2020.OL ADS.OL AMSC.OL AGAS.OL ALNG.OL BRUT.OL BWLPG.OL CLCO.OL EWIND.OL EQVA.OL FLNG.OL FRO.OL GOGL.OL HAFNI.OL HAV.OL HKY.OL HSHP.OL HAUTO.OL JIN.OL"
    stocks += " KCC.OL MPCC.OL ODF.OL ODFB.OL OET.OL PHLY.OL SAGA.OL STST.OL STSU.OL SNI.OL WAVI.OL WEST.OL"
    return stocks.split(" ")

def date_plus_offset(date: str, offset: str) -> str:
    normalized = normalize_period_string(offset)
    match = re.match(r"(\d+)([MWYD])", normalized)
    if not match:
        raise ValueError(f"Invalid offset string: {offset}")
    num, unit = int(match.group(1)), match.group(2)
    # Map to DateOffset keyword
    kwargs = {}
    if unit == "Y":
        kwargs["years"] = num
    elif unit == "M":
        kwargs["months"] = num
    elif unit == "W":
        kwargs["weeks"] = num
    elif unit == "D":
        kwargs["days"] = num
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    new_date = pd.to_datetime(date) + pd.DateOffset(**kwargs)
    return str(new_date.date())

class Simulation(yf.Tickers):
    def __init__(self, 
                 universe: list, 
                 start: str = None, 
                 end: str = None,
                 reweighting_frequency: str = None,
                 reweighting_strategy: str = None,
                 reinvestment_strategy: str = None,
                 risk_free: float = None,
                ):
        # parent class
        super().__init__(" ".join(universe))
        
        # public attributes
        self.universe = universe
        self.start = start
        self.end = end
        self.reweighting_frequency = reweighting_frequency
        self.reweighting_strategy = reweighting_strategy
        self.reinvestment_strategy = reinvestment_strategy
        self.risk_free = risk_free
        
        # hidden attributes
        self.history = None # pd.DataFrame
        self.information = None # pd.DataFrame
        self.base_currency = "USD" # string
        self.dividend_tax_rate = None # float
        self.reinvestment_loss_rate = None # float
        self.temp = None # anytype

    def get_history(self, interval="1d", to_base_currency: bool = False, offset: str ="1y") -> pd.DataFrame:
        if not isinstance(interval, str):
            raise ValueError("Interval must be a string.")
        if not interval in ["1d", "1mo", "1y"]:
            raise ValueError(f"Interval must be any of the following: {["1d", "1mo", "1y"]}.")
        if not offset in ["1w", "1mo", "3mo", "6mo", "1y", "3y"]:
            raise ValueError(f"Offset must be any of the following: {["1w", "1mo", "3mo", "6mo", "1y", "3y"]}.")
        if not isinstance(to_base_currency, bool):
            raise ValueError("The \'to_base_currency\' parameter must be boolean.")
        if self.start is None or self.end is None:
            raise ValueError("Please define both start date and end date.")

        # we recompute the start date
        start_date = date_minus_offset(self.start, offset)

        history = super().history(None, interval=interval, start=start_date, end=self.end).copy()
        history = history.bfill().ffill()
        if to_base_currency:
            currencies_dict = {}
            for stock in self.universe:
                currencies_dict[stock] = yf.Ticker(stock).info["currency"]
            forex_rates = {}
            for unique_currency in list(set(currencies_dict.values())):
                if self.base_currency == "USD":
                    try:
                        forex_rates[unique_currency] = yf.Ticker(f"{unique_currency}=X").history(None, interval=interval, start=start_date, end=self.end)["Open"]
                    except Exception as e:
                        raise ValueError(f"Problem fetching foreign exchange rate between {unique_currency} and {base_currency}.")
                elif self.base_currency == "EUR":
                    try:
                        if unique_currency != "USD":
                            forex_rates[unique_currency] = yf.Ticker(f"{self.base_currency}{unique_currency}=X").history(None, interval=interval, start=start_date, end=self.end)["Open"]
                        else:
                            forex_rates[unique_currency] = yf.Ticker("EUR=X").history(None, interval=interval, start=start_date, end=self.end)["Open"]
                    except:
                        raise ValueError(f"Problem fetching foreign exchange rate between {unique_currency} and {base_currency}.")
                else:
                    raise NotImplementedError("Foreign exchange rates for base currencies other than \'EUR\' or \'USD\' not implemented.")
            for metric, stock in history.columns:
                if not metric in ['Close', 'Dividends', 'High', 'Low', 'Open']:
                    history.loc[(metric, stock)] = history.loc[(metric, stock)] * forex_rates[currency_dict[stock]]
            history = history.bfill().ffill()

        # add risk-free asset
        idx = history.index
        rf_annual = self.risk_free if self.risk_free is not None else 0.
        days_diff = idx.to_series().diff().dt.days.fillna(1).values
        cumulative_rf = np.cumprod((1 + rf_annual) ** (days_diff / 365.0))
        metrics = ["Open", "Close", "Low", "High", "Volume", "Dividends"]
        columns = pd.MultiIndex.from_product([metrics, ["Risk Free"]])
        zeros = np.zeros_like(cumulative_rf)
        rf_data = np.vstack([
            cumulative_rf,  # Open
            cumulative_rf,  # Close
            cumulative_rf,  # Low
            cumulative_rf,  # High
            zeros, # Volume
            zeros  # Dividends
        ]).T

        rf_df = pd.DataFrame(rf_data, index=idx, columns=columns)
        history = pd.concat([history, rf_df], axis=1)

        # remove columns that failed to load
        cols_to_drop = []
        for col in history.columns:
            if (history[col].isna()).sum() > len(history) * 0.9:
                cols_to_drop.append(col)
        print(cols_to_drop)
        
        
        # save data
        history = history.drop(columns=cols_to_drop)
        self.universe = list(history.columns.get_level_values(1).unique())
        print(self.universe)
        self.history = history
        return history
    
    def get_returns(self, logarithmic=False, adjusted=False) -> pd.DataFrame:
        if not isinstance(logarithmic, bool):
            raise ValueError("The \'logarithmic\' parameter must be boolean.")
        if not isinstance(adjusted, bool):
            raise ValueError("The \'adjusted\' parameter must be boolean.")
        if self.history is None:
            warning.warn("Detected no historic rates. Calling \'get_history()\'.")
            self.get_history()
        closes = self.history["Close"]
        dividends = self.history["Dividends"].fillna(0.)
        if adjusted:
            # total return price series: P_t + D_t
            numerators = closes.add(dividends, fill_value=0.)
        else:
            numerators = closes
        denominators = closes.shift(1)
        if logarithmic:
            returns = np.log(numerators / denominators)
        else:
            returns = (numerators - denominators) / denominators
        returns = returns.fillna(0.)
        
        top_header = "Log returns" if logarithmic else "Returns"
        top_header = "Adj " + top_header if adjusted else top_header
        returns.columns = pd.MultiIndex.from_product([[top_header], returns.columns])
        self.history = pd.concat([self.history, returns], axis=1)
        return returns

    def get_information(self, to_base_currency: bool = False) -> pd.DataFrame:
        all_years = set()
        metrics = ["Book Equity", "EPS", "Shares"]
        
        universe = self.universe.copy()
        if "Risk Free" in universe:
            universe.remove("Risk Free")
        info = {}
        
        for asset in universe:
            asset_ticker = yf.Ticker(asset)
            
            # Get balance sheet
            try:
                balance_sheet = asset_ticker.balance_sheet
            except Exception as e:
                balance_sheet = None
                warnings.warn(f"Failed to load balance sheet of {asset}.")
                
            # Book Equity extraction
            stockholders = pd.Series(dtype=float)
            if balance_sheet is not None and not balance_sheet.empty:
                try:
                    stockholders = balance_sheet.loc["Stockholders Equity"]
                except Exception:
                    try:
                        stockholders = balance_sheet.loc["Common Stock Equity"]
                    except Exception:
                        stockholders = pd.Series(dtype=float)
                        warnings.warn(f"Failed to load equity for {asset}.")
                years = [pd.to_datetime(date).year for date in stockholders.index]
                stockholders.index = years
                all_years.update(years)
                # Shares extraction
                try:
                    ordinary = balance_sheet.loc["Ordinary Shares Number"]
                except Exception:
                    try:
                        ordinary = balance_sheet.loc["Shares Issued"]
                    except Exception:
                        ordinary = pd.Series(dtype=float)
                        warnigs.warn(f"Failed to load shares for {asset}.")
                years = [pd.to_datetime(date).year for date in ordinary.index]
                ordinary.index = years
                all_years.update(years)
            info[("Book Equity", asset)] = stockholders
            info[("Shares", asset)] = ordinary 
    
            # Get income statement
            try:
                income_statement = asset_ticker.income_stmt
            except Exception as e:
                income_statement = None
                warnings.warn(f"Failed to load income statement of {asset}.")
                    
            # EPS extraction
            eps = pd.Series(dtype=float)
            if income_statement is not None and not income_statement.empty:
                try:
                    eps = income_statement.loc["Diluted EPS"]
                except Exception:
                    try:
                        eps = income_statement.loc["Basic EPS"]
                    except Exception:
                        eps = pd.Series(dtype=float)
                        warnings.warn(f"Failed to load EPS for {asset}.")
                years = [pd.to_datetime(date).year for date in eps.index]
                eps.index = years
                all_years.update(years)
            info[("EPS", asset)] = eps
    
        all_years = sorted(all_years)
        datetime_index = pd.to_datetime([str(year) + "-01-01" for year in all_years])
        columns = pd.MultiIndex.from_product([metrics, universe])
        information = pd.DataFrame(index=datetime_index, columns=columns, dtype=float)
        for asset in universe:
            for metric in metrics:
                asset_metric = info.get((metric, asset), pd.Series(dtype=float))
                information[(metric, asset)] = information.index.year.map(asset_metric.get)


        for idx in information.index:
            # convert to UTC tz
            if getattr(idx, 'tzinfo', None) is None or idx.tzinfo is None:
                idx_dt = pd.Timestamp(idx).tz_localize('UTC')
            else:
                idx_dt = idx.tz_convert('UTC')
            
            # start_date = column - 1w; end_date = column + 1w
            start_date = date_minus_offset(str(idx_dt.date()), "1w")
            end_date = date_plus_offset(str(idx_dt.date()), "1w")
            
            if to_base_currency and self.base_currency == "USD":
                currencies_dict = {}
                for stock in universe:
                    currencies_dict[stock] = yf.Ticker(stock).info["currency"]
                forex_rates = {}
                for unique_currency in list(set(currencies_dict.values())):
                    if unique_currency not in forex_rates.keys():
                        try:
                            forex_series = yf.Ticker(f"{unique_currency}=X").history(None, interval="1d", start=start_date, end=end_date)["Open"]

                            # convert forex_series to UTC tz
                            if forex_series.index.tz is None:
                                forex_series.index = fx_series.index.tz_localize('UTC')

                            delta = forex_series.index - idx_dt
                            abs_delta = np.abs(delta) 
                            closest_idx = abs_delta.argmin()
                            closest_date = forex_series.index[closest_idx]
                            forex_rates[unique_currency] = forex_series.loc[closest_date]
                            
                        except Exception as e:
                            raise ValueError(f"Problem fetching foreign exchange rate between {unique_currency} and {self.base_currency}.")
                for metric, stock in product(["Book Equity", "EPS"], universe):
                    information.loc[idx, (metric, stock)] = information.loc[idx, (metric, stock)] * forex_rates[currencies_dict[stock]]
                print(f"Successfully converted information to {self.base_currency}")
                    
                    
            elif to_base_currency and self.base_currency != "USD":
                raise NotImplementedError(f"Cannot convert information to {self.base_currency} yet.")
            else:
                None
            
        self.information = information
        return information

        
        
    ##################
    # Public Setters #
    ##################

    def set_universe(self, universe: list):
        self.universe = universe
        super().__init__(" ".join(universe))  # re-call yf.Tickers init

    def set_start_and_end(self, start: str, end: str):
        # check dateformat: "YYYY-MM-DD"
        if start is None or end is None:
            raise ValueError("Both start and end dates must be provided.")
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid start date format: '{start}', expected 'YYYY-MM-DD'")
        try:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid end date format: '{end}', expected 'YYYY-MM-DD'")   
        # enforce chronological order
        if start_dt > end_dt:
            raise ValueError(f"Start date {start} cannot be after end date {end}.")
        self.end = end
        self.start = start

    def set_reweighting_frequency(self, freq: str):
        self.reweighting_frequency = freq

    def set_reweighting_strategy(self, strategy: str):
        if strategy == "uniform":
            self.reweighting_strategy = self.reweighting_uniform
        elif strategy == "relative_to_market_cap":
            self.reweighting_strategy = self.reweighting_relative_to_market_cap
        elif strategy == "small_low":
            self.reweighting_strategy = lambda date: self.reweighting_fama_and_french(date, False, "low")
        elif strategy == "small_medium":
            self.reweighting_strategy = lambda date: self.reweighting_fama_and_french(date, False, "medium")
        elif strategy == "small_high":
            self.reweighting_strategy = lambda date: self.reweighting_fama_and_french(date, False, "high")
        elif strategy == "big_low":
            self.reweighting_strategy = lambda date: self.reweighting_fama_and_french(date, True, "low")
        elif strategy == "big_medium":
            self.reweighting_strategy = lambda date: self.reweighting_fama_and_french(date, True, "medium")
        elif strategy == "big_high":
            self.reweighting_strategy = lambda date: self.reweighting_fama_and_french(date, True, "high")
        elif strategy == "test":
            self.reweighting_strategy = self.reweighting_test
        else:
            raise ValueError(f"The reweighting strategy {strategy} is not implemented.")

    def set_reinvestment_strategy(self, strategy: str):
        if strategy == "save_until_next_period":
            self.reinvestment_strategy = self.reinvestment_save_until_next_period
        elif strategy == "uniform":
            self.reinvestment_strategy = self.reinvestment_uniform
        elif strategy == "use_reweighting":
            self.reinvestment_strategy = self.reinvestment_by_temp
        else:
            raise ValueError(f"The rienvestment strategy {strategy} is not implemented.")
        
    def set_risk_free(self, rf: float):
        if rf is not None and not isinstance(rf, (int, float)):
            raise ValueError("Must set a numeric or None risk free rate.")
        if isinstance(rf, (int, float)) and (np.abs(rf) > 1):
            raise ValueError("Risk free rate must be in [0,1] if numeric.")
        self.risk_free = float(rf) if rf is not None else None
        
    ##################
    # Hidden Setters #
    ##################
    
    def set_history(self, history: pd.DataFrame):
        self.history = history

    def set_information(self, information: pd.DataFrame):
        self.information = information

    def set_base_currency(self, currency: str):
        if not re.match(r"^[A-Z]{3}$", currency):
            raise ValueError("Currency must be a 3-letter code, e.g. 'USD'")
        self.base_currency = currency

    def set_dividend_tax_rate(self, rate: float):
        if rate is not None and not isinstance(rate, (int, float)):
            raise ValueError("Must set a numeric or None dividend tax rate.")
        if isinstance(rate, (int, float)) and (rate > 1) or rate < 0:
            raise ValueError("Dividend tax rate must be in [0,1] if numeric.")
        self.dividend_tax_rate = float(rate) if rate is not None else None
        
    def set_reinvestment_loss_rate(self, loss: float):
        if loss is not None and not isinstance(loss, (int, float)):
            raise ValueError("Must set a numeric or None reinvestment loss rate.")
        if isinstance(loss, (int, float)) and (loss > 1) or loss < 0:
            raise ValueError("Reinvestment loss rate must be in [0,1] if numeric.")
        self.reinvestment_loss_rate = float(loss) if loss is not None else None

    
    ##########################
    # Reweighting strategies #
    ##########################

    def reweighting_relative_to_market_cap(self, date: str) -> pd.Series:
        date = pd.to_datetime(date)
        year = date.date().year - 1
        market_caps = self.information.loc[str(year)+"-01-01"]["Shares"] * self.history.iloc[-1]["Close"]
        weights = (market_caps/market_caps.sum()).fillna(0)
        return weights
        
    def reweighting_uniform(self, date: str) -> pd.Series:
        n = len(self.universe)
        weights = np.ones(n)/n
        return pd.Series(index=self.universe, data=weights)

    def reweighting_fama_and_french(self, date: str, upper_median: bool = False, book_to_market_level: str = "low") -> pd.Series:
        if book_to_market_level not in ["low", "medium", "high"]:
            raise ValueError(f"Book to market must be in {['low', 'medium', 'high']}.")
        date = pd.to_datetime(date)
        year = date.date().year - 1
        market_caps = self.information.loc[str(year)+"-01-01"]["Shares"] * self.history.iloc[-1]["Close"]
        books = self.information.loc[str(year)+"-01-01"]["Book Equity"] 
        book_to_market = books / market_caps
        
        # fama and french
        market_caps_median = np.nanmedian(market_caps)
        p30 = np.nanpercentile(book_to_market, 30)
        p70 = np.nanpercentile(book_to_market, 70)

        print(f"median: {market_caps_median:.3E}")
        print(f"p30: {p30:.3E}")
        print(f"p70: {p70:.3E}")
        
        if upper_median:
            selection = [asset for asset in self.universe if asset in market_caps.index and market_caps[asset] > market_caps_median]
        else:
            selection = [asset for asset in self.universe if asset in market_caps.index and market_caps[asset] <= market_caps_median]
        if book_to_market_level == "low":
            selection2 = [asset for asset in selection if book_to_market[asset] < p30]
        elif book_to_market_level == "medium":
            selection2 = [asset for asset in selection if book_to_market[asset] >= p30 and book_to_market[asset] < p70]
        else:
            selection2 = [asset for asset in selection if book_to_market[asset] >= p70]

        selection = list(set(selection).intersection(set(selection2)))
        
        weights = {}
        for asset in self.universe:
            if asset in selection and not math.isnan(market_caps[asset]):
                weights[asset] = market_caps[asset]
            else:
                weights[asset] = 0.
        
        weights = pd.Series(data=weights)
        sum_of_weights = weights.sum()
        if sum_of_weights == 0:
            raise ValueError("Sum of weights is zero: check asset selection criteria.")
        weights = weights/sum_of_weights
        assert np.isclose(weights.sum(), 1), "Weights do not sum to 1!"
        return weights
        
    # def reweighting_fama_and_french(self, history: pd.DataFrame, info: pd.DataFrame, upper_median: bool = False, book_to_market: str = "low") -> dict:
    #     if book_to_market not in ["low", "medium", "high"]:
    #         raise ValueError(f"Book to market must be in {['low', 'medium', 'high']}.")
        
    #     last_date = history.index[-1]
    #     if not isinstance(last_date, str):
    #         last_date = str(last_date.date())

    #     if self.book_to_market is not None and self.book_to_market["date"] == last_date:
    #         book_to_market = self.book_to_market["values"]
    #         # existence of one implies existence of other
    #         market_cap = self.market_cap["values"]
    #     else:
    #         new_start_date = date_minus_offset(last_date, self.reweighting_frequency)
    #         last_date = pd.to_datetime(last_date)
    
    #         universe = self.universe.copy()
    #         universe.remove("Risk Free")
    #         book_to_market = {}
    #         market_cap = {}
    #         for asset in universe:
    #             try:
    #                 # none of this should fail, if it does then stop the simulation
    #                 # we compute market cap as last close times avg. volume last period
    #                 market_cap[asset] = history.loc[last_date, ("Close", asset)] * history.loc[(new_start_date <= history.index) & (history.index <= last_date),("Volume", asset)].mean()
    #                 # we extract book equity last fiscal year, i.e. last year
    #                 if not info.index[info.index < last_date].empty:
    #                     book_ind = info.index[info.index < last_date][-1]
    #                     book = info.loc[book_ind, ("Book Equity", asset)]
    #             except Exception as e:
    #                 raise ValueError(f"Failed to fetch market cap or book equity for {asset}.")
                
    #             if math.isnan(market_cap[asset]) or math.isnan(book):
    #                 warnings.warn(f"NaN problem with market cap or book equity of {asset}. Skipping.")
    #                 book_to_market[asset] = np.nan
    #                 pass
    #             if market_cap[asset] <= 0 or book <= 0:
    #                 warnings.warn(f"Zero problem with market cap or book equity of {asset}. Skipping.")
    #                 book_to_market[asset] = np.nan
    #                 pass
                    
    #             book_to_market[asset] = book/market_cap[asset]
            
    #         send_forward_dict = {}
    #         send_forward_dict["date"] = last_date
    #         send_forward_dict["values"] = book_to_market
    #         self.book_to_market = send_forward_dict
    #         send_forward_dict_2 = {}
    #         send_forward_dict_2["date"] = last_date
    #         send_forward_dict_2["values"] = market_cap
    #         self.market_cap = send_forward_dict_2

    #     market_cap_median = np.median([v for v in book_to_market.values() if not math.isnan(v)])

    #     if upper_median:
    #         selected_assets = [asset for asset in market_cap.keys() if market_cap[asset] > 0 and market_cap[asset] > market_cap_median ]
    #     else:
    #         selected_assets = [asset for asset in market_cap.keys() if market_cap[asset] > 0 and market_cap[asset] < market_cap_median ]
        
    #     p30 = np.nanpercentile(list(book_to_market.values()), 30)
    #     p70 = np.nanpercentile(list(book_to_market.values()), 70)
        
    #     if book_to_market == "low":
    #         selected_assets_2 = [asset for asset in selected_assets if book_to_market[asset] < p30]
    #     elif book_to_market == "medium":
    #         selected_assets_2 = [asset for asset in selected_assets if book_to_market[asset] >= p30 and book_to_market[asset] < p70]
    #     else:
    #         selected_assets_2 = [asset for asset in selected_assets if book_to_market[asset] >= p70]

    #     selected_assets = list(set(selected_assets).intersection(set(selected_assets_2)))

    #     print(f"p30: {p30:.2E}, p70: {p70:.2E}, list= {selected_assets}")
    #     total_market_cap = sum([market_cap[asset] for asset in selected_assets])
    #     strategy = {}
    #     universe.append("Risk Free")
    #     for asset in universe:
    #         if asset in selected_assets and total_market_cap > 0:
    #             strategy[asset] = market_cap[asset] / total_market_cap
    #         else:
    #             strategy[asset] = 0.0
    
    #     return strategy
        
                

    # def reweighting_by_smallest_market_cap(self, history: pd.DataFrame, info: pd.DataFrame) -> dict:
    #     last_date = history.index[-1]
    #     if not isinstance(last_date, str):
    #         last_date = str(last_date.date())
    #     new_start_date = date_minus_offset(last_date, self.reweighting_frequency)
    #     new_start_date = pd.to_datetime(new_start_date)
    #     last_date = pd.to_datetime(last_date)
        
    #     market_cap = {}
    #     for asset in self.universe:
    #         market_cap[asset] = history.loc[last_date, ("Close", asset)] * history.loc[(new_start_date <= history.index) & (history.index <= last_date), ("Volume", asset)].mean()
    #     total_market = sum(market_cap.values())
        
    #     weights = {asset: market_cap[asset] / total_market for asset in self.universe if asset != "Risk Free"}
        
    #     # Inverse weights for smallest cap strategy
    #     inv_weights = {asset: 1.0 / w if w > 0 else 0.0 for asset, w in weights.items()}
    #     total_inv_weight = sum(inv_weights.values())
        
    #     strategy = {}
    #     for asset in self.universe:
    #         if asset != "Risk Free":
    #             strategy[asset] = inv_weights[asset] / total_inv_weight if total_inv_weight > 0 else 0.0
    #             if strategy[asset] < 0. or strategy[asset] > 1.:
    #                 raise ValueError(f"Computed weight for {asset} is {strategy[asset]:.2f}, which is invalid.")
    #         else:
    #             strategy[asset] = 0.0
    
    #     return strategy


    # def reweighting_test(self, history: pd.DataFrame, info: pd.DataFrame) -> dict:
    #     print("\nhistory:\n-----------------")
    #     print(history)
    #     print("\ninfo:\n------------------")
    #     print(info)
    #     return self.reweighting_uniform(history, info)
        

            


    ###########################
    # Reinvestment strategies #
    ###########################

    def reinvestment_save_until_next_period(self, date: str) -> pd.Series:
        weights = {}
        for asset in self.universe:
            weights[asset] = 0. if asset!="Risk Free" else 1.
        return pd.Series(data=weights)

    
    def reinvestment_uniform(self, date: str) -> pd.Series:
        n = len(self.universe)
        weights = np.ones(n)/n
        return pd.Series(index=self.universe, data=weights)

    def reinvestment_by_temp(self, date: str) -> pd.Series:
        if self.temp is not None and isinstance(self.temp, pd.Series):
            return self.temp
        else:
            data = np.ones(len(self.universe))
            return pd.Series(data, dtype=float)
            
    ##############
    # Simulation #
    ##############
    
    
    def simulate_one_period(self, initial_wealth: float, start, end) -> pd.DataFrame:
        """
        Simulates one period with reweighting_strategy and reinvestment_strategy
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        simulation_index = self.history.index[(start <= self.history.index) & (self.history.index <= end)]

        universe = self.universe
        reinvestment_loss_rate = 0. if self.reinvestment_loss_rate is None else self.reinvestment_loss_rate
        dividend_tax_rate = 0. if self.dividend_tax_rate is None else self.dividend_tax_rate
        weights = self.reweighting_strategy(start)
        self.temp = weights.copy()
        first_date = simulation_index[0]
        initial_prices = self.history.loc[first_date]["Open"]

        # we extract simulation history
        simulation_history = self.history.loc[simulation_index, :]
        
        for asset in universe:
            simulation_history[("Nb of assets", asset)] = 0

        for asset in universe:
            simulation_history.loc[first_date, ("Nb of assets", asset)] = weights[asset] * initial_wealth * (1 - reinvestment_loss_rate) / initial_prices[asset]
        
        
        simulation_history.loc[first_date, ("Wealth", "Portfolio")] = initial_wealth * (1 - reinvestment_loss_rate)
        simulation_history.loc[first_date, ("Wealth", "Total dividend")] = 0
        # simulate
        for i, idx in enumerate(simulation_index[1:]):

            last_row = simulation_history.loc[simulation_index[i]]
            row = simulation_history.loc[idx]

            total_dividend = 0
            for asset in universe:
                total_dividend += row.loc[("Dividends", asset)] * last_row[("Nb of assets", asset)]
            simulation_history.loc[idx, ("Wealth", "Total dividend")] = total_dividend            

            
            if total_dividend > 0:
                # we figure out how to spread dividend
                investments = self.reinvestment_strategy(str(idx.date()))
                if investments.sum() == 0:
                    raise ValueError(f"Potential divide by zero in \'simulate_one_period\'. Exiting...")
                else:
                    investments = investments/investments.sum()
                print(f"Reinvesting total dividend of {total_dividend:.3E} at {idx}")
                print(f"Weights: {investments}")
                for asset in universe:
                    current_price = row.loc[("Open", asset)]
                    investments[asset] *= total_dividend
                    current_nb_of_stocks = last_row.loc[("Nb of assets", asset)]
                    new_nb_of_stocks = current_nb_of_stocks + investments[asset] / current_price
                    
                    # we reinvest, i.e. update the nb of stocks
                    simulation_history.loc[idx, ("Nb of assets", asset)] = new_nb_of_stocks
            else:
                for asset in universe:
                    simulation_history.loc[idx, ("Nb of assets", asset)] = last_row[("Nb of assets", asset)]

            # we update wealth
            current_wealth = 0
            for asset in universe:
                current_wealth += simulation_history.loc[idx, ("Close", asset)] * simulation_history.loc[idx, ("Nb of assets", asset)]
                
            simulation_history.loc[idx, ("Wealth", "Portfolio")] = current_wealth

        return simulation_history

    def run(self, initial_wealth: float) -> pd.DataFrame:
        frequency = "1y" if self.reweighting_frequency is None else self.reweighting_frequency 

        if frequency not in ["1w", "1mo", "3mo", "1y"]:
            raise ValueError(f"Reweighting frequency must be one of {["1w", "1mo", "3mo", "1y"]}")

        universe = self.universe    
        history = self.history.copy()
        history.index = pd.to_datetime(history.index)

        # create columns for simulation
        for col in list(product(["Nb of assets"], universe)) + [("Wealth", "Total dividend"), ("Wealth", "Portfolio")]:
            history[col] = 0.
        
        
        start = pd.to_datetime(self.start)
        end_of_simulation = pd.to_datetime(self.end)
        simulation_dates = history[(start <= history.index) & (history.index <= end_of_simulation)].index
        i = 0
        
        while start < end_of_simulation:

            if i == 0:
                current_wealth = initial_wealth
            else:
                end_in_column = history.index[history.index <= end][-1]
                current_wealth = history.loc[end_in_column, ("Wealth", "Portfolio")]

            end = date_plus_offset(str(start.date()), frequency)
            end = pd.to_datetime(end)
            period_result = self.simulate_one_period(initial_wealth=current_wealth, start=start, end=min(end, end_of_simulation))

            history.loc[period_result.index, period_result.columns] = period_result

            # update
            start = history.index[history.index > end]
            start = end if start.empty else start[0]
            i += 1


        self.history = history
        return history.loc[simulation_dates, :]


