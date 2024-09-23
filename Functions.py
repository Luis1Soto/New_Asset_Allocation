import yfinance as yf
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as sco
import numpy as np
import matplotlib.pyplot as plt


class LoadData:
    def __init__(self, folder_path):
        """
        Initialize the LoadData object with the given folder path.
        """
        self.folder_path = folder_path
        self.financial_dataframes = {}
        self.process_excel_files()
        self.get_financial_dataframes()

    def harmonic_mean(self,values):
        values = np.array(values, dtype=float)
        non_zero_values = values[values != 0]  # Excluir los ceros
        if len(non_zero_values) == 0:
            return np.nan  # Devolver NaN si todos los valores son cero
        return len(non_zero_values) / np.sum(1.0 / non_zero_values)
        
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
                financial_data['Harmonic Mean'] = financial_data.iloc[:, 1:].apply(self.harmonic_mean, axis=1)
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

        # Define the financial ratios and their weights
        self.ratios = ['Return on Common Equity', 'Operating Margin', 'Current Ratio (x)', 
                       'Total Debt/Equity (%)', 'Price/Earnings', 'Cash Flow per Share']
        self.weights = {'Return on Common Equity': 0.30, 'Operating Margin': 0.20, 
                        'Current Ratio (x)': 0.15, 'Total Debt/Equity (%)': 0.15, 
                        'Price/Earnings': 0.10, 'Cash Flow per Share': 0.10}

    def calculate_monthly_returns(self, price_series):
        """
        Calculate the monthly returns for a given price series.
        
        Parameters:
        - price_series (pd.Series): A time series of stock prices.
        
        Returns:
        pd.Series: A time series of monthly returns.
        """
        return price_series.resample('ME').last().pct_change()

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
                    else:
                        print(f"No se encontraron datos financieros recientes para {ticker}")
                else:
                    print(f"No se encontraron datos financieros para {ticker}")
            
            if not df.empty:
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna()
                
                if df.empty:
                    print("No hay datos suficientes después de limpiar los datos.")
                    return []
                
                scaler = MinMaxScaler()
                normalized_ratios = pd.DataFrame(scaler.fit_transform(df), columns=self.ratios, index=df.index)
    
                df['Puntuacion_Total'] = 0
                for ratio in self.ratios:
                    df['Puntuacion_Total'] += normalized_ratios[ratio] * self.weights[ratio]
    
                # Seleccionar las 8 mejores acciones
                top_8 = df.nlargest(8, 'Puntuacion_Total').index.tolist()
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



class Backtesting:
    def __init__(self, results, prices, initial_capital):
        """
        Initialize the Backtesting object with strategy results, asset prices, and initial capital.
        
        Parameters:
        - results (pd.DataFrame): DataFrame containing the results of the strategy (run_strategy output).
        - prices (dict): Dictionary of tickers with their corresponding price data.
        - initial_capital (float): Initial capital for the portfolio.
        """
        self.results = results
        self.prices = prices
        self.initial_capital = initial_capital

    def rebalance(self, date, selected_stocks, universe_type, method="equal_weight"):
        """
        Rebalance the portfolio on the given date based on the selected stocks and rebalancing method.
        
        Parameters:
        - date (pd.Timestamp): The date of the rebalancing.
        - selected_stocks (list): List of selected tickers for rebalancing.
        - universe_type (str): Type of universe, either 'Defensive' or 'Offensive'.
        - method (str): Method of asset allocation optimization (equal_weight, sharpe, omega, etc.)
        
        Returns:
        dict: Dictionary containing the weights assigned to each asset.
        """
        if universe_type == "Defensive":
            # Defensive universe: equal weight allocation
            weights = np.ones(len(selected_stocks)) / len(selected_stocks)
        
        elif universe_type == "Offensive":
            # Offensive universe: use optimization methods
            if method == "equal_weight":
                weights = np.ones(len(selected_stocks)) / len(selected_stocks)
            elif method == "sharpe":
                weights = self.optimize_sharpe_ratio(selected_stocks)
            elif method == "omega":
                weights = self.optimize_omega_ratio(selected_stocks)
            elif method == "sortino":
                weights = self.optimize_sortino_ratio(selected_stocks)
            elif method == "cvar":
                weights = self.optimize_cvar(selected_stocks)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
        
        # Si la optimización falla o devuelve None, asignar pesos iguales
        if weights is None or not np.any(weights):
            weights = np.ones(len(selected_stocks)) / len(selected_stocks)

        return dict(zip(selected_stocks, weights))
    def run_backtest(self):
        """
        Run the backtest for multiple portfolio optimization strategies (CVaR, Sharpe, Sortino, etc.)
        
        Returns:
        pd.DataFrame: DataFrame containing the portfolio values over time for each strategy.
        """
        portfolio_values = {
            "CVaR": self.initial_capital,
            "Sharpe": self.initial_capital,
            "Sortino": self.initial_capital,
            "Equal Weight": self.initial_capital
        }
    
        # Historial del valor de los portafolios
        portfolio_history = []
    
        # Ajustar las fechas del DataFrame results a 'YYYY-MM-01'
        self.results['Date'] = pd.to_datetime(self.results['Date'], format='%Y-%m').dt.to_period('M').dt.to_timestamp()
    
        # Obtener todas las fechas disponibles en los precios de cada ticker
        all_dates = np.unique(np.concatenate([self.prices[ticker].index.values for ticker in self.prices.keys()]))
        all_dates = sorted(pd.to_datetime(all_dates))
    
        # Inicializar un diccionario para rastrear la cantidad de acciones compradas en cada estrategia
        holdings = {
            "CVaR": {},
            "Sharpe": {},
            "Sortino": {},
            "Equal Weight": {}
        }
    
        # Iterar sobre las fechas de los resultados (fechas de rebalanceo)
        for i in range(len(self.results)):
            row = self.results.iloc[i]
            date = row['Date']
            selected_stocks = row['Selected Stocks']
            universe = row['Chosen Universe']
    
            # Rebalancear cada estrategia de optimización
            weights_cvar = self.rebalance(date, selected_stocks, universe, method="cvar")
            weights_sharpe = self.rebalance(date, selected_stocks, universe, method="sharpe")
            weights_sortino = self.rebalance(date, selected_stocks, universe, method="sortino")
            weights_equal = self.rebalance(date, selected_stocks, universe, method="equal_weight")
    
            # Comprar acciones en el día de rebalanceo para cada estrategia
            self.buy_stocks(holdings, "CVaR", weights_cvar, portfolio_values["CVaR"], date)
            self.buy_stocks(holdings, "Sharpe", weights_sharpe, portfolio_values["Sharpe"], date)
            self.buy_stocks(holdings, "Sortino", weights_sortino, portfolio_values["Sortino"], date)
            self.buy_stocks(holdings, "Equal Weight", weights_equal, portfolio_values["Equal Weight"], date)
    
            # Desde la fecha de rebalanceo actual hasta la siguiente
            next_date = self.results.iloc[i + 1]['Date'] if i + 1 < len(self.results) else all_dates[-1]
            daily_dates = [d for d in all_dates if date <= d <= next_date]
    
            for current_date in daily_dates:
                portfolio_values["CVaR"] = self.calculate_portfolio_value(holdings["CVaR"], current_date)
                portfolio_values["Sharpe"] = self.calculate_portfolio_value(holdings["Sharpe"], current_date)
                portfolio_values["Sortino"] = self.calculate_portfolio_value(holdings["Sortino"], current_date)
                portfolio_values["Equal Weight"] = self.calculate_portfolio_value(holdings["Equal Weight"], current_date)
    
                # Registrar el valor de los portafolios en el historial
                portfolio_history.append({
                    "Date": current_date,
                    "CVaR": portfolio_values["CVaR"],
                    "Sharpe": portfolio_values["Sharpe"],
                    "Sortino": portfolio_values["Sortino"],
                    "Equal Weight": portfolio_values["Equal Weight"]
                })
    
        return pd.DataFrame(portfolio_history)
    
    def buy_stocks(self, holdings, strategy, weights, portfolio_value, date):
        """
        Comprar acciones en el día del rebalanceo en base a los pesos asignados.
        
        Parameters:
        - holdings (dict): Diccionario que almacena la cantidad de acciones compradas por estrategia.
        - strategy (str): El nombre de la estrategia (CVaR, Sharpe, Sortino, Equal Weight).
        - weights (dict): Pesos asignados a cada activo.
        - portfolio_value (float): Valor total del portafolio en la fecha del rebalanceo.
        - date (pd.Timestamp): La fecha en la que se realiza el rebalanceo.
        """
        for ticker, weight in weights.items():
            if ticker in self.prices and date in self.prices[ticker].index:
                stock_price = self.prices[ticker].loc[date]
                amount_to_invest = portfolio_value * weight
                # Comprar la cantidad de acciones correspondiente
                holdings[strategy][ticker] = amount_to_invest / stock_price
    
    def calculate_portfolio_value(self, holdings, current_date):
        """
        Calcular el valor total del portafolio basado en las acciones poseídas y los precios actuales.
        
        Parameters:
        - holdings (dict): Cantidad de acciones poseídas de cada activo.
        - current_date (pd.Timestamp): La fecha actual.
        
        Returns:
        float: Valor actualizado del portafolio.
        """
        total_value = 0
        for ticker, num_shares in holdings.items():
            if ticker in self.prices and current_date in self.prices[ticker].index:
                stock_price = self.prices[ticker].loc[current_date]
                total_value += num_shares * stock_price
        return total_value
    
    def update_portfolio_value(self, portfolio_value, weights, current_date):
        """
        Update the portfolio value based on the current asset prices.
        
        Parameters:
        - portfolio_value (float): The current value of the portfolio.
        - weights (dict): Dictionary of asset weights.
        - current_date (pd.Timestamp): The date for which to update the portfolio value.
        
        Returns:
        float: Updated portfolio value.
        """
        new_value = 0
        for ticker, weight in weights.items():
            if ticker in self.prices:
                price_data = self.prices[ticker]
                if current_date in price_data.index:
                    current_price = price_data.loc[current_date]
                    previous_price = price_data.loc[price_data.index[0]]  # Precio inicial de rebalanceo
                    asset_return = current_price / previous_price - 1
                    new_value += weight * (portfolio_value * (1 + asset_return))
        return new_value

    def plot_portfolio_evolution(self, portfolio_history):
        """
        Plot the evolution of portfolio values over time for different strategies.
        
        Parameters:
        - portfolio_history (pd.DataFrame): DataFrame containing the portfolio values over time.
        """
        plt.figure(figsize=(10, 6))
        for strategy in ["CVaR", "Sharpe", "Sortino", "Equal Weight"]:
            plt.plot(portfolio_history['Date'], portfolio_history[strategy], label=strategy)

        plt.title("Portfolio Evolution Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def get_asset_returns(self, selected_stocks):
        """
        Retrieve historical returns for the selected stocks.
        
        Parameters:
        - selected_stocks (list): List of selected tickers.
        
        Returns:
        pd.DataFrame: A DataFrame containing historical returns for the selected assets.
        """
        asset_returns = pd.DataFrame()
    
        for ticker in selected_stocks:
            if ticker in self.prices:
                returns = self.prices[ticker].pct_change().dropna()
                asset_returns[ticker] = returns
        
        return asset_returns

    def get_portfolio_return(self, weights, date):
        """
        Calculate the portfolio return based on the weights and asset returns.
        
        Parameters:
        - weights (dict): Dictionary of asset weights.
        - date (pd.Timestamp): Date for which to calculate the returns.
        
        Returns:
        float: Portfolio return since the last rebalancing.
        """
        portfolio_return = 0
        for ticker, weight in weights.items():
            if ticker in self.prices:
                price_data = self.prices[ticker]
                if date in price_data.index:
                    previous_price = price_data.loc[date]
                    current_price = price_data.iloc[-1]
                    asset_return = (current_price / previous_price) - 1
                    portfolio_return += weight * asset_return
        return portfolio_return

    def optimize_sharpe_ratio(self, selected_stocks):
        """
        Optimize the portfolio to maximize the Sharpe ratio.
        """
        def neg_sharpe_ratio(weights):
            asset_returns = self.get_asset_returns(selected_stocks)
            portfolio_return = np.dot(weights, asset_returns.mean())
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(asset_returns.cov(), weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility
            return -sharpe_ratio  
    
        num_assets = len(selected_stocks)
        start_weights = np.ones(num_assets) / num_assets
        bounds = [(0, 1) for _ in range(num_assets)]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        result = sco.minimize(neg_sharpe_ratio, start_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else None
    
    
    def optimize_omega_ratio(self, selected_stocks):
        """
        Optimize the portfolio to maximize the Omega ratio.
        """
        def neg_omega_ratio(weights):
            # Obtener los retornos de los activos
            asset_returns = self.get_asset_returns(selected_stocks)
            portfolio_returns = np.dot(weights, asset_returns.T)
            
            threshold = 0
            gains = portfolio_returns[portfolio_returns > threshold].sum()
            losses = -portfolio_returns[portfolio_returns < threshold].sum()
            omega_ratio = gains / losses if losses > 0 else float('inf')
            
            return -omega_ratio
    
        num_assets = len(selected_stocks)
        start_weights = np.ones(num_assets) / num_assets
        bounds = [(0, 1) for _ in range(num_assets)]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        result = sco.minimize(neg_omega_ratio, start_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else None
    
    def optimize_sortino_ratio(self, selected_stocks):
        """
        Optimize the portfolio to maximize the Sortino ratio.
        """
        def neg_sortino_ratio(weights):
            asset_returns = self.get_asset_returns(selected_stocks)
            portfolio_return = np.dot(weights, asset_returns.mean())
            
            downside_risk = np.sqrt(np.mean(np.square(np.minimum(np.dot(weights, asset_returns.T), 0))))
            sortino_ratio = portfolio_return / downside_risk if downside_risk > 0 else float('inf')
            
            return -sortino_ratio
    
        # Optimización
        num_assets = len(selected_stocks)
        start_weights = np.ones(num_assets) / num_assets
        bounds = [(0, 1) for _ in range(num_assets)]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        result = sco.minimize(neg_sortino_ratio, start_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else None
    
    def optimize_cvar(self, selected_stocks, alpha=0.05):
        """
        Optimize the portfolio to minimize Conditional Value at Risk (CVaR).
        """
        def cvar(weights):
            asset_returns = self.get_asset_returns(selected_stocks)
            portfolio_returns = np.dot(weights, asset_returns.T)
            
            VaR = np.percentile(portfolio_returns, alpha * 100)
            CVaR = portfolio_returns[portfolio_returns <= VaR].mean()
            
            return -CVaR  # Minimizar CVaR
    
        num_assets = len(selected_stocks)
        start_weights = np.ones(num_assets) / num_assets
        bounds = [(0, 1) for _ in range(num_assets)]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        result = sco.minimize(cvar, start_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else None

    def plot_portfolio_evolution(self, portfolio_history):
        """
        Plot the evolution of portfolio values over time for different strategies.
        
        Parameters:
        - portfolio_history (pd.DataFrame): DataFrame containing the portfolio values over time.
        """
        plt.figure(figsize=(10, 6))
        for strategy in ["CVaR", "Sharpe", "Sortino", "Equal Weight"]:
            plt.plot(portfolio_history['Date'], portfolio_history[strategy], label=strategy)

        plt.title("Portfolio Evolution Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.show()


    

























































