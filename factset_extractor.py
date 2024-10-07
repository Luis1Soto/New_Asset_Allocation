import json
import pandas as pd

from types import SimpleNamespace


with open("./data.json", "r") as file:
    content = file.read()

data = json.loads(content, object_hook=lambda d: SimpleNamespace(**d))

data_dict = {}

rowData = data.ksFinancialsTableFull.rowData

for i in range(90):
    row = getattr(rowData, f"row{i}")
    name = getattr(row, "col-0").value
    historical = list(map(lambda x: None if x == "@NA" else float(x),
                          getattr(row, "col-1").value.split(",")))
    data_dict[name] = historical

columnData = data.ksFinancialsTableFull.columnData
index = [getattr(columnData, f"col-{i}").value[0] for i in range(2, 27)]

dataframe = pd.DataFrame.from_dict(data_dict)
dataframe["Date"] = index[::-1]
dataframe.set_index("Date", inplace=True)

print(dataframe)

# dataframe.to_csv("fundamentals.csv")
