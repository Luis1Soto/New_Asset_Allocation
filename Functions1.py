import yfinance as yf
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as sco
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import warnings


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
        from datetime import datetime
    
        def convert_to_date_format(date_str):
            """
            Convierte una fecha en formato 'MMM 'YY' a 'YYYY-MM'.
            """
            return datetime.strptime(date_str, "%b '%y").strftime("%Y-%m")
    
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.xlsx'):  
                file_path = os.path.join(self.folder_path, file_name)
                excel_data = pd.ExcelFile(file_path)
                first_sheet_df = excel_data.parse(excel_data.sheet_names[0])
                ticker = first_sheet_df.iloc[4, 0].split()[0]
                dates = first_sheet_df.iloc[6, :].values
                dates = [convert_to_date_format(date) if isinstance(date, str) and "'" in date else date for date in dates]
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
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not stock_data.empty:
                closing_prices[ticker] = stock_data['Close']

        return closing_prices, self.financial_dataframes


    def beta(self, start_date='2020-01-01', end_date=None, market_index="SPY"):
    
        defensive_universe = []
        offensive_universe = []

        market_data = yf.download(market_index, start=start_date, end=end_date, progress=False)
        market_returns = market_data['Close'].pct_change().dropna()

        # Descargar precios y calcular betas
        for ticker, data in self.Load(start_date, end_date)[0].items():
            stock_returns = data.pct_change().dropna()

            # Encontrar la intersección de fechas
            common_dates = market_returns.index.intersection(stock_returns.index)
            stock_returns = stock_returns.loc[common_dates]
            market_returns_aligned = market_returns.loc[common_dates]

            # Asegurarse de que las longitudes de los datos coincidan
            if len(stock_returns) == len(market_returns_aligned):
                # Calcular covarianza y varianza para obtener beta
                covariance = np.cov(stock_returns, market_returns_aligned)[0, 1]
                variance = market_returns_aligned.var()
                beta = covariance / variance

                # Clasificar en universos
                if beta <= 0.6:
                    defensive_universe.append(ticker)
                else:
                    offensive_universe.append(ticker)
            else:
                print(f"Datos faltantes para {ticker}, no se incluye en el cálculo de beta.")

        return {"defensive_universe": defensive_universe, "offensive_universe": offensive_universe}

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

        # Define the financial ratios and their weights
        self.ratios = ['Return on Common Equity', 'Operating Margin', 'Current Ratio (x)', 
                       'Total Debt/Equity (%)', 'Price/Earnings', 'Cash Flow per Share','Total Shares Outstanding  (M)']


    def calculate_monthly_returns(self, price_series):
        """
        Calculate the monthly returns for a given price series.
        
        Parameters:
        - price_series (pd.Series): A time series of stock prices.
        
        Returns:
        pd.Series: A time series of monthly returns.
        """
        return price_series.resample('ME').last().pct_change()


    def calculate_harmonic_mean(self, ratios):
        return len(ratios) / np.sum(1.0 / ratios)

    def calculate_piotroski_score(self, row):
        score = 0
        if row['Return on Common Equity'] > 0:
            score += 1
        if row['Return on Common Equity'] > np.roll(row['Return on Common Equity'], shift=1):
            score += 1
        if row['Operating Margin'] > 0:
            score += 1
        if row['Operating Margin'] >  np.roll(row['Operating Margin'], shift=1):
            score += 1
        if row['Cash Flow per Share'] > 0:
            score += 1
        if row['Current Ratio (x)'] > np.roll(row['Current Ratio (x)'], shift=1):
            score += 1
        if row['Total Debt/Equity (%)'] < np.roll(row['Total Debt/Equity (%)'], shift=1):
            score += 1
        if row['Total Shares Outstanding  (M)'] <= np.roll(row['Total Shares Outstanding  (M)'], shift=1):
            score += 1
        if row['Price/Earnings'] < 15:
            score += 2
        return score

   
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
                price_data = self.prices[ticker].resample('ME').last()
                returns = price_data.pct_change()
                
                if end_date:
                    returns = returns.loc[:end_date]

                if len(returns) >= window:
                    sma = returns.rolling(window=window).mean()
                    momentum[ticker] = sma.iloc[-1]

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

        avg_momentum = pd.Series(momentum_data).mean()

        momentum_data['Average'] = avg_momentum
        self.momentum_data = pd.concat([self.momentum_data, pd.DataFrame(momentum_data, index=[date])])

        if avg_momentum > 0:
            return "Offensive"
        else:
            return "Defensive"

    def select_top_stocks(self, universe, decision_date):
        """
        Selecciona las mejores acciones basadas en los ratios financieros más recientes en la fecha de decisión.
        
        Parameters:
        - universe (list): Lista de tickers en el universo seleccionado (Offensive o Defensive).
        - decision_date (str o pd.Timestamp): Fecha de la decisión del universo.
        
        Returns:
        list: Las mejores 8 acciones si el universo es Offensive, o todas las acciones si es Defensive.
        """
        decision_date = pd.to_datetime(decision_date)
    
        if universe == "Offensive":
            df = pd.DataFrame()
    
            for ticker in self.offensive:
                if ticker in self.financials:
                    
                    financial_data = self.financials[ticker]
                    
                    report_dates = pd.to_datetime(financial_data.columns, format='%Y-%m', errors='coerce')
                    
                    date_mapping = dict(zip(report_dates, financial_data.columns))
                    
                    valid_dates = [dt for dt in report_dates if dt <= decision_date and not pd.isnull(dt)]
                    
                    if valid_dates:
                        most_recent_date = max(valid_dates)
                        most_recent_column = date_mapping[most_recent_date]
                        most_recent_data = financial_data[['Financial Ratio', most_recent_column]]
                        
                        ratios_data = most_recent_data.set_index('Financial Ratio').loc[self.ratios][most_recent_column]
                        ratios_data.name = ticker
                        df = pd.concat([df, pd.DataFrame([ratios_data])])
            
            if not df.empty:
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna()
                
                if df.empty:
                    print("No hay datos suficientes después de limpiar los datos.")
                    return []

                df['Harmonic_Mean'] = df.apply(lambda row: self.calculate_harmonic_mean(row) if all(row > 0) else 0, axis=1)
                df['Piotroski_Score'] = df.apply(self.calculate_piotroski_score, axis=1)
                top_8 = df.nlargest(8, 'Piotroski_Score').index.tolist()
                
                return top_8
            else:
                print("No hay datos suficientes para seleccionar acciones.")
                return []
        
        elif universe == "Defensive":
            return self.defensive
    
        
    def run_strategy(self, start_date, end_date=None):
        """
        Run the strategy, making universe decisions every 6 months, starting from a year after start_date.
        
        Parameters:
        - start_date (str or pd.Timestamp): The start date of the strategy.
        - end_date (str or pd.Timestamp): The end date of the strategy. If None, use the most recent date in the data.
        
        Returns:
        pd.DataFrame: A DataFrame with the date, chosen universe, and selected stocks for each 6-month period.
        """
        start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = max([self.prices[ticker].index[-1] for ticker in self.protective])

        decision_date = start_date + pd.DateOffset(months=12)
        results = []

        while decision_date <= end_date:
            chosen_universe = self.decide_universe(decision_date)
            selected_stocks = self.select_top_stocks(chosen_universe, decision_date)

            results.append({
                'Date': decision_date,
                'Chosen Universe': chosen_universe,
                'Selected Stocks': selected_stocks
            })

            decision_date += pd.DateOffset(months=6)

        return pd.DataFrame(results)

    def get_momentum_dataframe(self):
        """
        Get a DataFrame containing the momentum values for each decision date.
        
        Returns:
        pd.DataFrame: DataFrame containing the momentum of protective tickers over time.
        """
        return self.momentum_data


class DynamicBacktest:
    
    def __init__(self, results, prices, initial_capital, benchmark_data=None, benchmark_ticker='^GSPC'):
        """
        Inicializa la clase con los parámetros dados y descarga el benchmark solo si no se proporciona.
        
        Args:
        - results: DataFrame con columnas ['Date', 'Chosen Universe', 'Selected Stocks']
        - prices: Diccionario con tickers como claves y series de pandas con precios como valores.
        - initial_capital: Capital inicial para el portafolio.
        - benchmark_data: Serie de tiempo opcional con los precios del benchmark.
        - benchmark_ticker: Ticker del benchmark para descargar si no se proporciona data.
        """
        self.results = results
        self.prices = prices
        self.initial_capital = initial_capital
        self.portfolio_values_sortino = []
        self.portfolio_values_benchmark = []
        self.weights_history_sortino = []

        self.benchmark_data = benchmark_data if benchmark_data is not None else self.download_benchmark_data(benchmark_ticker)
        self.benchmark_shares = None

        warnings.filterwarnings("ignore", category=RuntimeWarning, 
                                message="Values in x were outside bounds during a minimize step, clipping to bounds")

        self.run_backtest()

    def download_benchmark_data(self, ticker):
        """
        Descarga los datos históricos del benchmark usando yfinance.
        """
        benchmark_df = yf.download(ticker, start=self.results['Date'].min(), end=self.results['Date'].max(), progress=False)
        return benchmark_df['Adj Close']

    def calculate_sortino_weights(self, selected_stocks, end_date):
        """
        Calcula los pesos óptimos basados en el ratio Sortino.
        """
        start_date = end_date - pd.DateOffset(days=365)
        returns = pd.DataFrame({
            ticker: self.prices[ticker].loc[start_date:end_date].pct_change().dropna() 
            for ticker in selected_stocks if ticker in self.prices
        }).dropna(axis=1)

        def sortino_ratio(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            downside_std = np.sqrt(np.sum((returns[returns < 0].fillna(0).mean() * weights) ** 2) * 252)
            return -portfolio_return / downside_std if downside_std != 0 else np.inf

        n = len(returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0.05, 1) for _ in range(n))
        initial_weights = n * [1. / n,]

        optimized = sco.minimize(sortino_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return {ticker: optimized.x[i] for i, ticker in enumerate(returns.columns)}

    def calculate_semivariance_weights(self, selected_stocks, end_date):
        """
        Calcula los pesos óptimos minimizando la semivarianza del portafolio.
        """
        start_date = end_date - pd.DateOffset(days=365)
        returns = pd.DataFrame({
            ticker: self.prices[ticker].loc[start_date:end_date].pct_change().dropna()
            for ticker in selected_stocks if ticker in self.prices
        }).dropna(axis=1)
    
        def semivariance_loss(weights):
            portfolio_return = np.dot(returns, weights)
            downside_returns = portfolio_return[portfolio_return < 0]
            semivariance = np.mean(downside_returns ** 2)  # Semivarianza
            return semivariance
    
        n = len(returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0.05, 1) for _ in range(n))
        initial_weights = n * [1. / n,]
    
        optimized = sco.minimize(semivariance_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return {ticker: optimized.x[i] for i, ticker in enumerate(returns.columns)}
    
    def rebalance_portfolios(self, capital, selected_stocks, universe_type, date):
        """
        Realiza el rebalanceo utilizando Sortino para ofensivo y semivarianza para defensivo.
        """
        if universe_type == 'Offensive':
            weights = self.calculate_sortino_weights(selected_stocks, date)
        else:
            weights = self.calculate_semivariance_weights(selected_stocks, date)
    
        prices_at_rebalance = {ticker: self.prices[ticker].asof(date) for ticker in selected_stocks}
        shares = {ticker: (capital * weights[ticker]) / prices_at_rebalance[ticker] for ticker in selected_stocks}
        return shares, weights

    
    def run_backtest(self):
        """
        Ejecuta el backtest, manejando portafolios y benchmark.
        """
        capital = self.initial_capital
        rebalance_dates = pd.to_datetime(self.results['Date'].tolist())
        start_date = rebalance_dates[0]
        selected_stocks = self.results.iloc[0]['Selected Stocks']
        universe_type = self.results.iloc[0]['Chosen Universe']

        shares, weights = self.rebalance_portfolios(capital, selected_stocks, universe_type, start_date)
        self.weights_history_sortino.append((start_date, weights))
        benchmark_initial_price = self.benchmark_data.asof(start_date)
        self.benchmark_shares = self.initial_capital / benchmark_initial_price

        all_dates = pd.date_range(start_date, self.benchmark_data.index[-1])
        for current_date in all_dates:
            if current_date in rebalance_dates:
                idx = rebalance_dates.get_loc(current_date)
                capital = self.calculate_portfolio_value(shares, selected_stocks, current_date)
                selected_stocks = self.results.iloc[idx]['Selected Stocks']
                universe_type = self.results.iloc[idx]['Chosen Universe']
                shares, weights = self.rebalance_portfolios(capital, selected_stocks, universe_type, current_date)
                self.weights_history_sortino.append((current_date, weights))

            daily_value = self.calculate_portfolio_value(shares, selected_stocks, current_date)
            daily_value_benchmark = self.benchmark_shares * self.benchmark_data.asof(current_date)

            self.portfolio_values_sortino.append((current_date, daily_value))
            self.portfolio_values_benchmark.append((current_date, daily_value_benchmark))

    
    def calculate_portfolio_value(self, shares, selected_stocks, date):
        """
        Calcula el valor del portafolio para la fecha dada.
        """
        return sum(shares.get(ticker, 0) * self.prices[ticker].asof(date) for ticker in selected_stocks)

    def get_portfolio_values(self):
        """
        Devuelve un DataFrame con los valores del portafolio y del benchmark.
        """
        sortino_df = pd.DataFrame(self.portfolio_values_sortino, columns=['Date', 'Sortino Portfolio Value'])
        benchmark_df = pd.DataFrame(self.portfolio_values_benchmark, columns=['Date', 'Benchmark Portfolio Value'])
        return sortino_df.merge(benchmark_df, on='Date')

    def plot_strategies(self):
        """
        Grafica la evolución de los portafolios y el benchmark.
        """
        portfolio_values_df = self.get_portfolio_values()
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_values_df['Date'], portfolio_values_df['Sortino Portfolio Value'], label='Sortino Portfolio')
        plt.plot(portfolio_values_df['Date'], portfolio_values_df['Benchmark Portfolio Value'], label='Benchmark', color='red', linewidth=2)
        plt.title('Evolución del Portafolio y Benchmark')
        plt.xlabel('Fecha')
        plt.ylabel('Valor del Portafolio')
        plt.legend()
        plt.grid()
        plt.show()

    def evaluate_portfolios(self):
        """
        Calcula y devuelve un DataFrame con métricas de rendimiento para los portafolios Sortino y Semivarianza, 
        junto con el benchmark.
    
        Returns:
        - DataFrame con métricas de rendimiento: 'mean_return', 'CAGR', 'Sharpe Ratio', 
          'Sortino Ratio', 'Semivariance', 'Volatility', 'VaR', 'Beta', 'Max Drawdown', 'Tracking Error', 'Alpha'.
        """
        portfolio_values_df = self.get_portfolio_values()
        returns_df = portfolio_values_df.set_index('Date').pct_change().dropna()
    
        sortino_returns = returns_df['Sortino Portfolio Value']
        benchmark_returns = returns_df['Benchmark Portfolio Value']
    
        def calculate_cagr(portfolio_values):
            initial_value = portfolio_values.iloc[0]
            final_value = portfolio_values.iloc[-1]
            total_periods = len(portfolio_values)
            years = total_periods / 252  
            cagr = (final_value / initial_value) ** (1 / years) - 1
            return cagr
    
        def max_drawdown(portfolio_values):
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - running_max) / running_max
            return drawdown.min()
    
        def calculate_beta(portfolio_returns, market_returns):
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance
    
        def calculate_semivariance(returns):
            downside_returns = returns[returns < 0]
            return np.mean(downside_returns ** 2) * 252
    
        sortino_metrics = {
            'mean_return': sortino_returns.mean() * 252,
            'CAGR': calculate_cagr(portfolio_values_df['Sortino Portfolio Value']),
            'Sharpe Ratio': sortino_returns.mean() / sortino_returns.std() * np.sqrt(252),
            'Sortino Ratio': sortino_returns.mean() / sortino_returns[sortino_returns < 0].std() * np.sqrt(252),
            'Volatility': sortino_returns.std() * np.sqrt(252),
            'Semivariance': calculate_semivariance(sortino_returns),
            'VaR (5%)': np.percentile(sortino_returns, 5),
            'Beta': calculate_beta(sortino_returns, benchmark_returns),
            'Max Drawdown': max_drawdown(portfolio_values_df['Sortino Portfolio Value']),
            'Tracking Error': np.sqrt(np.mean((sortino_returns - benchmark_returns) ** 2)) * np.sqrt(252),
            'Alpha': sortino_returns.mean() * 252 - calculate_beta(sortino_returns, benchmark_returns) * benchmark_returns.mean() * 252
        }
    
        benchmark_metrics = {
            'mean_return': benchmark_returns.mean() * 252,
            'CAGR': calculate_cagr(portfolio_values_df['Benchmark Portfolio Value']),
            'Volatility': benchmark_returns.std() * np.sqrt(252),
            'Max Drawdown': max_drawdown(portfolio_values_df['Benchmark Portfolio Value'])
        }
    
        metrics_df = pd.DataFrame([sortino_metrics, benchmark_metrics], index=['Sortino', 'Benchmark'])
        return metrics_df