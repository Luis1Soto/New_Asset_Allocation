import yfinance as yf
import pandas as pd




class AssetDataFetcher:
    def __init__(self, offensive_tickers, defensive_tickers, protective_tickers):
        self.groups = {
            'Offensive': offensive_tickers,
            'Defensive': defensive_tickers,
            'Protective': protective_tickers
        }        
        self.all_tickers = offensive_tickers + defensive_tickers + protective_tickers
        self.data = {}

    def fetch_data(self):
        for ticker in self.all_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                ratios = self.extract_ratios(info)
                self.data[ticker] = ratios
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

    def extract_ratios(self, info):
        ratios = {
            'P/E Ratio': info.get('trailingPE'),
            'P/BV Ratio': info.get('priceToBook'),
            'P/S Ratio': info.get('priceToSalesTrailing12Months'),
            'Value/EBIT Ratio': self.calculate_value_to_ebit(info),
            'Value/EBITDA Ratio': self.calculate_value_to_ebitda(info),
            'Value/Sales Ratio': self.calculate_value_to_sales(info),
            'Value/Book Capital Ratio': self.calculate_value_to_book_capital(info)
        }
        return ratios

    def calculate_value_to_ebit(self, info):
        ev = info.get('enterpriseValue')
        ebit = info.get('ebit')
        if ev is not None and ebit is not None and ebit != 0:
            return ev / ebit
        return 'N/A'

    def calculate_value_to_ebitda(self, info):
        ev = info.get('enterpriseValue')
        ebitda = info.get('ebitda')
        if ev is not None and ebitda is not None and ebitda != 0:
            return ev / ebitda
        return 'N/A'

    def calculate_value_to_sales(self, info):
        ev = info.get('enterpriseValue')
        sales = info.get('totalRevenue')
        if ev is not None and sales is not None and sales != 0:
            return ev / sales
        return 'N/A'

    def calculate_value_to_book_capital(self, info):
        ev = info.get('enterpriseValue')
        total_assets = info.get('totalAssets')
        total_liabilities = info.get('totalLiab')
        if ev is not None and total_assets is not None and total_liabilities is not None:
            book_cap = total_assets - total_liabilities
            if book_cap is not None and book_cap != 0:
                return ev / book_cap
        return 'N/A'

    def get_data(self):
        df = pd.DataFrame(self.data).T  
        return df




































































