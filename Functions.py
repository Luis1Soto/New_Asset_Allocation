import pandas as pd
import os



class LoadData:
    def __init__(self, folder_path):
        """
        Initialize the LoadData object with the given folder path.
        """
        self.folder_path = folder_path
        self.financial_dataframes = {}
        self.process_excel_files()

    
    def process_excel_files(self):
        """
        Process all Excel files in the given folder and extract financial ratios.
        """
        # Loop through all files in the folder
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.xlsx'):  # Only process Excel files
                file_path = os.path.join(self.folder_path, file_name)

                # Load Excel file
                excel_data = pd.ExcelFile(file_path)

                # Assuming we are interested in the first sheet
                first_sheet_df = excel_data.parse(excel_data.sheet_names[0])

                # Extract ticker (ticker is in the first column of row 4)
                ticker = first_sheet_df.iloc[4, 0].split()[0]

                # Extract dates from row 6 and use them as column headers for financial data
                dates = first_sheet_df.iloc[6, :].values

                # Extract the relevant financial data starting from row 11
                financial_data = first_sheet_df.iloc[10:, :].reset_index(drop=True)

                # Set the first column as the index (it contains the names of the financial ratios)
                financial_data.columns = dates
                financial_data.rename(columns={financial_data.columns[0]: 'Financial Ratio'}, inplace=True)

                # Store the dataframe in the dictionary with the ticker as the key
                self.financial_dataframes[ticker] = financial_data

        # Print a message with all tickers for which data was processed
        print(f"Data processed for the following tickers: {', '.join(self.financial_dataframes.keys())}")

    def get_financial_dataframes(self):
        """
        Return the dictionary of financial dataframes.
        """
        return self.financial_dataframes

# Example usage:
# loader = LoadData(r'C:\path\to\folder')
# loader.process_excel_files()
# financials = loader.get_financial_dataframes()





































































