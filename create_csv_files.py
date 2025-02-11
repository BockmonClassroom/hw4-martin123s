# 02/11/2025
# Shuiming Chen

# import pandas to print data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# get the Iris raw data 
data = pd.read_csv("./data/Iris.csv")

# 1. create an Min-Max Scaling
scaler_normalize = MinMaxScaler()

normalize = data.copy()

normalize[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]] = scaler_normalize.fit_transform(normalize[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]).round(4)

# turn it into csv file
normalize.to_csv('normalized_data.csv', index=False)


# 2. create an StandardScaler
scaler_standard = StandardScaler()

standardlize = data.copy()

standardlize[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]] = scaler_standard.fit_transform(standardlize[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]).round(4)

# turn it into csv file
standardlize.to_csv('standlized_data.csv', index=False)
