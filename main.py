from ai.nn import EthernumPriceNeuralNetwork
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/data.csv', index_col='Date')
df.drop(['High', 'Low', 'Mid', 'Bid', 'Ask', 'Volume'], inplace=True, axis=1)

scaler = MinMaxScaler()
data = scaler.fit_transform(df)

eth_nn = EthernumPriceNeuralNetwork()
eth_nn.train(
    data=data,
    model_name='./models/22_08_model'
)