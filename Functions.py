import yfinance as yf
import os
import pandas as pd

class LoadData:
    def __init__(self, folder_path):
        """
        Initialize the LoadData object with the given folder path.
        """
        self.folder_path = folder_path
        self.financial_dataframes = {}
        self.process_excel_files()
        self.get_financial_dataframes()
        
    def process_excel_files(self):
        """
        Process all Excel files in the given folder and extract financial ratios.
        """
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.xlsx'):  
                file_path = os.path.join(self.folder_path, file_name)
                excel_data = pd.ExcelFile(file_path)
                first_sheet_df = excel_data.parse(excel_data.sheet_names[0])
                ticker = first_sheet_df.iloc[4, 0].split()[0]
                dates = first_sheet_df.iloc[6, :].values
                financial_data = first_sheet_df.iloc[10:, :].reset_index(drop=True)
                financial_data.columns = dates
                financial_data.rename(columns={financial_data.columns[0]: 'Financial Ratio'}, inplace=True)
                self.financial_dataframes[ticker] = financial_data

        print(f"Data processed for the following tickers: {', '.join(self.financial_dataframes.keys())}")

    def get_financial_dataframes(self):
        """
        Return the dictionary of financial dataframes.
        """
        return self.financial_dataframes

    def Load(self, start_date='2020-01-01', end_date=None):
        """
        Download daily closing prices for the tickers processed.
        
        Parameters:
        start_date (str): The start date for downloading historical data (YYYY-MM-DD format).
        end_date (str): The end date for downloading historical data. If None, today's date is used.
        
        Returns:
        dict: A dictionary where keys are tickers and values are dataframes of daily closing prices.
        """
        closing_prices = {}

        for ticker in self.financial_dataframes.keys():
            print(f"Downloading data for {ticker}...")
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                closing_prices[ticker] = stock_data['Close']
            else:
                print(f"No data found for {ticker}.")

        return closing_prices, self.financial_dataframes


class TestStrategy:
    def __init__(self, prices, financials, offensive, defensive, protective):
        """
        Initialize the TestStrategy object with prices, financials, and ticker universes.
        
        Parameters:
        - prices (dict): Dictionary of tickers with their corresponding price data (from LoadData).
        - financials (dict): Dictionary of tickers with their corresponding financial data (from LoadData).
        - offensive (list): List of tickers in the Offensive universe.
        - defensive (list): List of tickers in the Defensive universe.
        - protective (list): List of tickers in the Protective universe.
        """
        self.prices = prices
        self.financials = financials
        self.offensive = offensive
        self.defensive = defensive
        self.protective = protective
        self.momentum_data = pd.DataFrame()  # Store momentum values for each decision date

    def calculate_monthly_returns(self, price_series):
        """
        Calculate the monthly returns for a given price series.
        
        Parameters:
        - price_series (pd.Series): A time series of stock prices.
        
        Returns:
        pd.Series: A time series of monthly returns.
        """
        return price_series.resample('M').last().pct_change()

    def get_momentum(self, window=12, end_date=None):
        """
        Calculate the momentum (SMA12) for tickers in the protective universe up to a given end date.
        
        Parameters:
        - window (int): The number of periods (months) for calculating the simple moving average (SMA).
        - end_date (str or pd.Timestamp): The date up to which to calculate the momentum. If None, use the latest available date.
        
        Returns:
        dict: A dictionary with tickers as keys and their momentum values as values.
        """
        momentum = {}

        for ticker in self.protective:
            if ticker in self.prices:
                # Calculate monthly returns
                price_data = self.prices[ticker].resample('M').last()
                returns = price_data.pct_change()
                
                # Filter the data up to the specified end_date
                if end_date:
                    returns = returns.loc[:end_date]

                if len(returns) >= window:
                    sma = returns.rolling(window=window).mean()
                    # Take the latest SMA value as momentum
                    momentum[ticker] = sma.iloc[-1]
                else:
                    print(f"Not enough data to calculate momentum for {ticker}")
            else:
                print(f"No price data available for {ticker}")

        return momentum

    def decide_universe(self, date):
        """
        Decide whether to choose the Defensive or Offensive universe based on the momentum of Protective tickers.
        
        Parameters:
        - date (str or pd.Timestamp): The date up to which to make the decision.
        
        Returns:
        str: 'Defensive' if the momentum of Protective tickers is negative, 'Offensive' if positive.
        """
        momentum_data = self.get_momentum(end_date=date)

        # Calculate absolute momentum (average momentum of all protective tickers)
        avg_momentum = pd.Series(momentum_data).mean()

        # Store momentum for reference
        momentum_data['Average'] = avg_momentum
        self.momentum_data = self.momentum_data = pd.concat([self.momentum_data, pd.DataFrame(momentum_data, index=[date])])

        if avg_momentum > 0:
            return "Offensive"
        else:
            return "Defensive"

    def run_strategy(self, start_date, end_date=None):
        """
        Run the strategy, making universe decisions every 6 months, starting from a year after start_date.
        
        Parameters:
        - start_date (str or pd.Timestamp): The start date of the strategy.
        - end_date (str or pd.Timestamp): The end date of the strategy. If None, use the most recent date in the data.
        
        Returns:
        pd.DataFrame: A DataFrame with the date and the chosen universe for each 6-month period.
        """
        # Convert start_date and end_date to Timestamps
        start_date = pd.to_datetime(start_date)
        if end_date is None:
            # Use the most recent date available in the price data for any protective ticker
            end_date = max([self.prices[ticker].index[-1] for ticker in self.protective])

        # Initialize the decision date to be 12 months after the start_date
        decision_date = start_date + pd.DateOffset(months=12)
        results = []

        while decision_date <= end_date:
            # Make a decision at the current decision_date
            chosen_universe = self.decide_universe(decision_date)
            results.append({
                'Date': decision_date,
                'Chosen Universe': chosen_universe
            })

            # Move forward 6 months for the next decision
            decision_date += pd.DateOffset(months=6)

        # Convert the results to a DataFrame
        return pd.DataFrame(results)

    def get_momentum_dataframe(self):
        """
        Get a DataFrame containing the momentum values for each decision date.
        
        Returns:
        pd.DataFrame: DataFrame containing the momentum of protective tickers over time.
        """
        return self.momentum_data

# Example usage:
# loader = LoadData(r'C:\path\to\folder')
# loader.process_excel_files()
# prices = loader.download_daily_closing_prices(start_date='2020-01-01')
# financials = loader.get_financial_dataframes()

# offensive_tickers = ['AAPL', 'MSFT', 'GOOGL']
# defensive_tickers = ['JNJ', 'PG', 'KO']
# protective_tickers = ['TLT', 'GLD', 'VIX']

# strategy = TestStrategy(prices, financials, offensive_tickers, defensive_tickers, protective_tickers)
# strategy_results = strategy.run_strategy(start_date='2020-01-01')
# momentum_df = strategy.get_momentum_dataframe()
# print(strategy_results)
# print(momentum_df)



































































