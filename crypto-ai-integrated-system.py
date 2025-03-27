import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from transformers import pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from web3 import Web3
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CryptoAIProject:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.nlp = pipeline("sentiment-analysis")
        self.web3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_INFURA_KEY'))
        self.models = {}
        
    # 1. Real-time Price Fetcher
    def get_crypto_price(self, coin='bitcoin'):
        url = f"{self.base_url}/simple/price?ids={coin}&vs_currencies=usd"
        return requests.get(url).json()
    
    # 2. Historical Data Analyzer
    def get_historical_data(self, coin='bitcoin', days=30):
        url = f"{self.base_url}/coins/{coin}/market_chart?vs_currency=usd&days={days}"
        return pd.DataFrame(requests.get(url).json()['prices'], columns=['timestamp', 'price'])
    
    # 3. Sentiment Analyzer
    def analyze_sentiment(self, text):
        return self.nlp(text)[0]
    
    # 4. Price Predictor (LSTM)
    def train_price_predictor(self, data, epochs=10):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60,1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        X, y = self._create_dataset(data, 60)
        model.fit(X, y, epochs=epochs)
        self.models['price_predictor'] = model
        return model
    
    # 5. Anomaly Detector
    def detect_anomalies(self, data):
        clf = IsolationForest(contamination=0.1)
        data['anomaly'] = clf.fit_predict(data[['price']])
        return data
    
    # 6. Trading Strategy Optimizer
    def optimize_strategy(self, data):
        data['returns'] = np.log(data['price'] / data['price'].shift(1))
        data['strategy'] = data['returns'].rolling(5).mean()
        return data.dropna()
    
    # 7. Wallet Generator
    def generate_wallet(self):
        account = self.web3.eth.account.create()
        return {'address': account.address, 'key': account.key.hex()}
    
    # 8. Transaction Simulator
    def simulate_transaction(self, sender, receiver, amount):
        return {
            'from': sender,
            'to': receiver,
            'value': amount,
            'gas': 21000,
            'gasPrice': self.web3.eth.gas_price,
            'nonce': 0
        }
    
    # 9. Blockchain Data Analyzer
    def analyze_block(self, block_num):
        block = self.web3.eth.get_block(block_num)
        return {
            'block_number': block.number,
            'transactions': len(block.transactions),
            'timestamp': datetime.fromtimestamp(block.timestamp)
        }
    
    # 10. News Aggregator
    def get_crypto_news(self):
        url = f"{self.base_url}/news"
        return [article['title'] for article in requests.get(url).json()[:5]]
    
    # 11. Social Media Monitor
    def monitor_social_media(self, query='bitcoin'):
        url = f"https://social-media-api.com/search?q={query}"
        return requests.get(url).json()  # Mock endpoint
    
    # 12. Risk Assessor
    def assess_risk(self, data):
        return np.std(data['price']) * np.sqrt(365)
    
    # 13. Arbitrage Finder
    def find_arbitrage(self, exchanges):
        prices = {ex: self.get_exchange_price(ex) for ex in exchanges}
        return max(prices, key=prices.get), min(prices, key=prices.get)
    
    # 14. Pattern Recognizer
    def recognize_patterns(self, data):
        data['MA50'] = data['price'].rolling(50).mean()
        data['MA200'] = data['price'].rolling(200).mean()
        return data
    
    # 15. ICO Analyzer
    def analyze_ico(self, contract_address):
        contract = self.web3.eth.contract(address=contract_address, abi=[])
        return {
            'total_supply': contract.functions.totalSupply().call(),
            'holders': contract.functions.balanceOf(contract_address).call()
        }
    
    # 16. DeFi Analytics
    def analyze_defi(self):
        return requests.get('https://api.defipulse.com/v1/statistics').json()
    
    # 17. NFT Analyzer
    def analyze_nft(self, contract_address):
        return self.web3.eth.get_transaction_count(contract_address)
    
    # 18. Market Trend Analyzer
    def analyze_trends(self, coins, days=30):
        trends = {}
        for coin in coins:
            data = self.get_historical_data(coin, days)
            trends[coin] = data['price'].pct_change().mean()
        return trends
    
    # 19. Volume Predictor
    def predict_volume(self, data):
        model = LinearRegression()
        X = np.arange(len(data)).reshape(-1,1)
        model.fit(X, data['volume'])
        return model.predict([[len(data)+1]])[0]
    
    # 20. Sentiment-Trend Correlation
    def analyze_sentiment_correlation(self, texts, prices):
        sentiments = [self.analyze_sentiment(text)['score'] for text in texts]
        return np.corrcoef(sentiments, prices)[0,1]
    
    # 21. AI Auditor
    def audit_contract(self, contract_code):
        detector = pipeline("text-classification", model="codebert-base")
        return detector(contract_code)
    
    # Helper function
    def _create_dataset(self, dataset, time_step=1):
        X, y = [], []
        for i in range(len(dataset)-time_step-1):
            X.append(dataset[i:(i+time_step), 0])
            y.append(dataset[i+time_step, 0])
        return np.array(X), np.array(y)

# Contoh Penggunaan
if __name__ == "__main__":
    project = CryptoAIProject()
    
    # 1. Dapatkan harga real-time
    print("Harga Bitcoin:", project.get_crypto_price()['bitcoin']['usd'])
    
    # 2. Analisis data historis
    btc_data = project.get_historical_data()
    
    # 3. Analisis sentimen
    print("Sentimen:", project.analyze_sentiment("Bitcoin akan naik!"))
    
    # 4. Prediksi harga
    model = project.train_price_predictor(btc_data[['price']].values)
    
    # 5. Deteksi anomali
    btc_data = project.detect_anomalies(btc_data)
    
    # 6. Optimalkan strategi trading
    strategy = project.optimize_strategy(btc_data)
    
    # 7. Generate wallet
    wallet = project.generate_wallet()
    print("Wallet Baru:", wallet)
    
    # 8. Simulasi transaksi
    tx = project.simulate_transaction(wallet['address'], '0x...', 1)
    
    # 9. Analisis blok
    block = project.analyze_block(10000000)
    
    # 10. Dapatkan berita
    news = project.get_crypto_news()
    
    # ... dan seterusnya untuk semua 21 fitur
