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
#            else:
#                print(f"No data found for {ticker}.")

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
#                else:
#                    print(f"Not enough data to calculate momentum for {ticker}")
#            else:
#                print(f"No price data available for {ticker}")

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
#                    else:
#                        print(f"No se encontraron datos financieros recientes para {ticker}")
#                else:
#                    print(f"No se encontraron datos financieros para {ticker}")
            
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

        # Convert the results to a DataFrame
        return pd.DataFrame(results)

    def get_momentum_dataframe(self):
        """
        Get a DataFrame containing the momentum values for each decision date.
        
        Returns:
        pd.DataFrame: DataFrame containing the momentum of protective tickers over time.
        """
        return self.momentum_data


class DynamicBacktest:
    def __init__(self, results, prices, initial_capital, benchmark_ticker='^GSPC'):
        """
        Inicializa la clase con los parámetros dados y descarga los datos del benchmark.
        
        Args:
        - results: DataFrame con columnas ['Date', 'Chosen Universe', 'Selected Stocks']
        - prices: Diccionario con tickers como claves y series de pandas con precios como valores.
        - initial_capital: Capital inicial para el portafolio.
        - benchmark_ticker: Ticker del benchmark para descargar (por defecto '^GSPC').
        """
        self.results = results
        self.prices = prices
        self.initial_capital = initial_capital
        self.portfolio_values_sortino = []  
        self.portfolio_values_sharpe = []  
        self.portfolio_values_cvar = []  
        self.portfolio_values_benchmark = []
        self.weights_history_sortino = []  
        self.weights_history_sharpe = []  
        self.weights_history_cvar = []  
        self.benchmark_data = self.download_benchmark_data(benchmark_ticker)
        self.benchmark_shares = None
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Values in x were outside bounds during a minimize step, clipping to bounds")
        self.run_backtest()
        
    def download_benchmark_data(self, ticker):
        """
        Descarga los datos históricos del benchmark utilizando yfinance.
        Args:
        - ticker: Ticker del benchmark (por defecto '^GSPC').
        Returns:
        - Serie de tiempo de los precios de cierre ajustados del benchmark.
        """
        benchmark_df = yf.download(ticker, start=self.results['Date'].min(), end=self.results['Date'].max(), progress=False)
        benchmark_series = benchmark_df['Adj Close']
        return benchmark_series

    
    def calculate_cvar_weights(self, selected_stocks, end_date, alpha=0.05):
        start_date = end_date - pd.DateOffset(days=365)
        filtered_prices = {ticker: self.prices[ticker].loc[start_date:end_date].dropna() for ticker in selected_stocks}
        filtered_prices = {ticker: prices for ticker, prices in filtered_prices.items() if len(prices) > 0}
        if len(filtered_prices) == 0:
            raise ValueError(f"No hay datos de precios disponibles en el rango {start_date} a {end_date} para los activos seleccionados.")
        returns = pd.DataFrame({ticker: prices.pct_change().dropna() for ticker, prices in filtered_prices.items()})
        
        def cvar_loss(weights):
            portfolio_returns = np.dot(returns, weights)
            sorted_returns = np.sort(portfolio_returns)
            cvar = -np.mean(sorted_returns[:int(alpha * len(sorted_returns))])
            return cvar
        
        n = len(returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0.05, 1) for asset in range(n))
        initial_weights = n * [1. / n,]
        optimized = sco.minimize(cvar_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_weights = optimized.x
        weights = {ticker: opt_weights[i] for i, ticker in enumerate(returns.columns)}
        return weights

    
    def calculate_sortino_weights(self, selected_stocks, end_date):
        start_date = end_date - pd.DateOffset(days=365)
        filtered_prices = {ticker: self.prices[ticker].loc[start_date:end_date].dropna() for ticker in selected_stocks}
        filtered_prices = {ticker: prices for ticker, prices in filtered_prices.items() if len(prices) > 0}
        if len(filtered_prices) == 0:
            raise ValueError(f"No hay datos de precios disponibles en el rango {start_date} a {end_date} para los activos seleccionados.")
        returns = pd.DataFrame({ticker: prices.pct_change().dropna() for ticker, prices in filtered_prices.items()})

        def sortino_ratio(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            downside_returns = returns[returns < 0].fillna(0)
            downside_std = np.sqrt(np.sum((downside_returns.mean() * weights) ** 2) * 252)
            if downside_std == 0:
                return np.inf
            sortino = portfolio_return / downside_std
            return -sortino

        n = len(returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0.05, 1) for asset in range(n))
        initial_weights = n * [1. / n,]
        optimized = sco.minimize(sortino_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_weights = optimized.x
        weights = {ticker: opt_weights[i] for i, ticker in enumerate(returns.columns)}
        return weights

    def calculate_sharpe_weights(self, selected_stocks, end_date):
        start_date = end_date - pd.DateOffset(days=365)
        filtered_prices = {ticker: self.prices[ticker].loc[start_date:end_date].dropna() for ticker in selected_stocks}
        filtered_prices = {ticker: prices for ticker, prices in filtered_prices.items() if len(prices) > 0}
        if len(filtered_prices) == 0:
            raise ValueError(f"No hay datos de precios disponibles en el rango {start_date} a {end_date} para los activos seleccionados.")
        returns = pd.DataFrame({ticker: prices.pct_change().dropna() for ticker, prices in filtered_prices.items()})

        def sharpe_ratio(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            if portfolio_volatility == 0:
                return np.inf
            return -(portfolio_return / portfolio_volatility)

        n = len(returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0.05, 1) for asset in range(n))
        initial_weights = n * [1. / n,]
        optimized = sco.minimize(sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_weights = optimized.x
        weights = {ticker: opt_weights[i] for i, ticker in enumerate(returns.columns)}
        return weights

    def rebalance_portfolios(self, capital, selected_stocks, universe_type, date):
        if universe_type == 'Offensive':
            weights_sortino = self.calculate_sortino_weights(selected_stocks, date)
            weights_sharpe = self.calculate_sharpe_weights(selected_stocks, date)
            weights_cvar = self.calculate_cvar_weights(selected_stocks, date)
        else:
            weights_sortino = {ticker: 1.0 / len(selected_stocks) for ticker in selected_stocks}
            weights_sharpe = {ticker: 1.0 / len(selected_stocks) for ticker in selected_stocks}
            weights_cvar = {ticker: 1.0 / len(selected_stocks) for ticker in selected_stocks}

        prices_at_rebalance = {ticker: self.prices[ticker].asof(date) for ticker in selected_stocks}
        shares_sortino = {ticker: (capital * weights_sortino[ticker]) / prices_at_rebalance[ticker] for ticker in selected_stocks}
        shares_sharpe = {ticker: (capital * weights_sharpe[ticker]) / prices_at_rebalance[ticker] for ticker in selected_stocks}
        shares_cvar = {ticker: (capital * weights_cvar[ticker]) / prices_at_rebalance[ticker] for ticker in selected_stocks}

        return shares_sortino, shares_sharpe, shares_cvar, weights_sortino, weights_sharpe, weights_cvar


    def run_backtest(self):
        capital_sortino = self.initial_capital
        capital_sharpe = self.initial_capital
        capital_cvar = self.initial_capital
        shares_sortino = {}
        shares_sharpe = {}
        shares_cvar = {}
        rebalance_dates = pd.to_datetime(self.results['Date'].tolist())

        start_date = rebalance_dates[0]
        selected_stocks = self.results.iloc[0]['Selected Stocks']
        universe_type = self.results.iloc[0]['Chosen Universe']
        shares_sortino, shares_sharpe, shares_cvar, weights_sortino, weights_sharpe, weights_cvar = self.rebalance_portfolios(capital_sortino, selected_stocks, universe_type, start_date)
        self.weights_history_sortino.append((start_date, weights_sortino))
        self.weights_history_sharpe.append((start_date, weights_sharpe))
        self.weights_history_cvar.append((start_date, weights_cvar))

        benchmark_initial_price = self.benchmark_data.asof(start_date)
        self.benchmark_shares = self.initial_capital / benchmark_initial_price

        all_dates = pd.date_range(start_date, self.prices[selected_stocks[0]].index[-1])

        for current_date in all_dates:
            if current_date in rebalance_dates:
                idx = rebalance_dates.get_loc(current_date)
                capital_sortino = self.calculate_portfolio_value(shares_sortino, selected_stocks, current_date)
                capital_sharpe = self.calculate_portfolio_value(shares_sharpe, selected_stocks, current_date)
                capital_cvar = self.calculate_portfolio_value(shares_cvar, selected_stocks, current_date)
                selected_stocks = self.results.iloc[idx]['Selected Stocks']
                universe_type = self.results.iloc[idx]['Chosen Universe']
                shares_sortino, shares_sharpe, shares_cvar, weights_sortino, weights_sharpe, weights_cvar = self.rebalance_portfolios(capital_sortino, selected_stocks, universe_type, current_date)
                self.weights_history_sortino.append((current_date, weights_sortino))
                self.weights_history_sharpe.append((current_date, weights_sharpe))
                self.weights_history_cvar.append((current_date, weights_cvar))

            daily_value_sortino = self.calculate_portfolio_value(shares_sortino, selected_stocks, current_date)
            daily_value_sharpe = self.calculate_portfolio_value(shares_sharpe, selected_stocks, current_date)
            daily_value_cvar = self.calculate_portfolio_value(shares_cvar, selected_stocks, current_date)
            
            daily_value_benchmark = self.benchmark_shares * self.benchmark_data.asof(current_date)
            
            self.portfolio_values_sortino.append((current_date, daily_value_sortino))
            self.portfolio_values_sharpe.append((current_date, daily_value_sharpe))
            self.portfolio_values_cvar.append((current_date, daily_value_cvar))
            self.portfolio_values_benchmark.append((current_date, daily_value_benchmark))

    
    def calculate_portfolio_value(self, shares, selected_stocks, date):
        portfolio_value = sum([shares[ticker] * self.prices[ticker].asof(date) for ticker in selected_stocks if ticker in shares])
        return portfolio_value

    def get_portfolio_values(self):
        sortino_df = pd.DataFrame(self.portfolio_values_sortino, columns=['Date', 'Sortino Portfolio Value'])
        sharpe_df = pd.DataFrame(self.portfolio_values_sharpe, columns=['Date', 'Sharpe Portfolio Value'])
        cvar_df = pd.DataFrame(self.portfolio_values_cvar, columns=['Date', 'CVaR Portfolio Value'])
        benchmark_df = pd.DataFrame(self.portfolio_values_benchmark, columns=['Date', 'benchmark Portfolio Value'])
        return sortino_df.merge(sharpe_df, on='Date').merge(cvar_df, on='Date').merge(benchmark_df, on='Date')

    def get_weights_history(self):
        sortino_weights = pd.DataFrame(self.weights_history_sortino, columns=['Date', 'Sortino Weights'])
        sharpe_weights = pd.DataFrame(self.weights_history_sharpe, columns=['Date', 'Sharpe Weights'])
        return sortino_weights.merge(sharpe_weights, on='Date', how='outer')

    def plot_strategies(self):
        portfolio_values_df = self.get_portfolio_values()
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_values_df['Date'], portfolio_values_df['Sortino Portfolio Value'], label='Sortino Portfolio')
        plt.plot(portfolio_values_df['Date'], portfolio_values_df['Sharpe Portfolio Value'], label='Sharpe Portfolio')
        plt.plot(portfolio_values_df['Date'], portfolio_values_df['CVaR Portfolio Value'], label='CVaR Portfolio')
        plt.plot(portfolio_values_df['Date'], portfolio_values_df['benchmark Portfolio Value'], label='Benchmark Portfolio', color='red', linewidth=2)
        plt.title('Evolución de los Portafolios')
        plt.xlabel('Fecha')
        plt.ylabel('Valor del Portafolio')
        plt.legend()
        plt.grid()



    def evaluate_portfolios(self):
        """
        Calcula y devuelve un DataFrame con las métricas de rendimiento para los portafolios disponibles (Sortino, Sharpe, benchmark).
        
        Returns:
        - DataFrame con métricas de rendimiento: 'mean_return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio',
          'Treynor Ratio', 'Volatility', 'VaR', 'Beta', 'Recovery Time', 'Tracking Error', 'Kurtosis', 'Max Drawdown'.
        """
        # Obtener los valores del portafolio y calcular retornos diarios
        portfolio_values_df = self.get_portfolio_values()
        returns_df = portfolio_values_df.set_index('Date').pct_change().dropna()

        # Calcular los retornos de cada portafolio
        sortino_returns = returns_df['Sortino Portfolio Value']
        sharpe_returns = returns_df['Sharpe Portfolio Value']
        cvar_returns = returns_df['CVaR Portfolio Value']
        benchmark_returns = returns_df['benchmark Portfolio Value']

        # Función para calcular CAGR
        def calculate_cagr(portfolio_values):
            # Validar si la serie tiene suficientes datos
            if len(portfolio_values) == 0:
                return np.nan
            # Acceder al primer y último valor usando .iloc
            initial_value = portfolio_values.iloc[0]
            final_value = portfolio_values.iloc[-1]
            total_periods = len(portfolio_values)
            years = total_periods / 252  # Asumiendo 252 días hábiles por año
            cagr = (final_value / initial_value) ** (1 / years) - 1
            return cagr

        # Función para calcular máxima caída (Max Drawdown)
        def max_drawdown(portfolio_values):
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - running_max) / running_max
            return drawdown.min()

        # Función para calcular tiempo de recuperación
        def recovery_time(portfolio_values):
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - running_max) / running_max
            drawdown_periods = np.where(drawdown < 0)[0]
            if len(drawdown_periods) == 0:
                return 0
            recovery_periods = drawdown_periods[-1] - drawdown_periods[0]
            return recovery_periods

        # Función para calcular beta del portafolio
        def calculate_beta(portfolio_returns, market_returns):
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance

        # Calcular métricas
        metrics = {}
        for name, returns, values in [('Sortino', sortino_returns, portfolio_values_df['Sortino Portfolio Value']),
                                      ('Sharpe', sharpe_returns, portfolio_values_df['Sharpe Portfolio Value']),
                                      ('CVaR', cvar_returns, portfolio_values_df['CVaR Portfolio Value']),
                                      ('benchmark', benchmark_returns, portfolio_values_df['benchmark Portfolio Value'])]:
            # Métricas individuales
            metrics[name] = {
                'mean_return': returns.mean() * 252,
                'CAGR': calculate_cagr(values),
                'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252),
                'Sortino Ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252),
                'Treynor Ratio': returns.mean() / calculate_beta(returns, benchmark_returns),
                'Volatility': returns.std() * np.sqrt(252),
                'VaR': np.percentile(returns, 5),
                'Beta': calculate_beta(returns, benchmark_returns),
                'Recovery Time': recovery_time(values),
                'Tracking Error': np.sqrt(np.mean((returns - benchmark_returns) ** 2)) * np.sqrt(252),
                'Kurtosis': kurtosis(returns),
                'Max Drawdown': max_drawdown(values),
                'Alpha': returns.mean() * 252 - calculate_beta(returns, benchmark_returns) * benchmark_returns.mean() * 252
            }

        metrics_df = pd.DataFrame(metrics).T
        return metrics_df