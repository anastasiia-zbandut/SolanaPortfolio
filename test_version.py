from seaborn import heatmap
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize


def price_data(from_symbol, to_symbol, exchange, datetime_interval):
    headers= {'authorization': f'Apikey e78d3dfbe4f59a6cb17877c8f5cb04b4bbcf943c59aa03ccc81dbeb90bcfaa09'}
    base_url = 'https://min-api.cryptocompare.com/data/histo'
    url = '%s%s' % (base_url, datetime_interval)
    params = {'fsym': from_symbol, 'tsym': to_symbol, 'limit': 2000, 'aggregate': 1, 'e': exchange}
    request = requests.get(url, headers=headers, params=params)
    data = request.json()
    return data


def convert_price_data(data):
    df = pd.json_normalize(data, ['Data'])
    df['datetime'] = pd.to_datetime(df.time, unit='s')
    df = df[['datetime', 'close']]
    return df


def coinmarketcap_info():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {'start':'1', 'limit':'1000', 'convert':'USD'}
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': '0f504583-1204-4b13-92b4-26df20c40db4'}
    request = requests.get(url, headers=headers, params=parameters)
    load = json.loads(request.text)
    data = pd.DataFrame(load['data'])
    data.drop(columns=['slug', 'self_reported_circulating_supply', 'self_reported_market_cap',
                       'last_updated', 'max_supply'], inplace=True)
    data['date_added'] = pd.to_datetime(data.date_added)
    data[['name', 'symbol']] = data[['name', 'symbol']].astype("string")
    quote = data['quote'].apply(pd.Series)['USD'].apply(pd.Series)
    quote.index = data['symbol']
    platform = data['platform']
    tags = data['tags']
    tags.index = data['symbol']
    data.drop(columns=['quote', 'platform', 'tags'], inplace=True)
    # columns = [[date.today(), date.today(), date.today()], ['name', 'symbol', 'cmc_rank']]
    # cmc_top_coins = pd.DataFrame(columns=pd.MultiIndex.from_arrays(columns))
    # for i in columns[1]:
    #     cmc_top_coins[date.today(), i] = data[i]
    columns = ['name', 'symbol', 'cmc_rank']
    cmc_top_coins = pd.DataFrame(columns=columns)
    for i in columns:
        cmc_top_coins[i] = data[i]
    return cmc_top_coins


cmc_top_coins = coinmarketcap_info()[0:9]


def check_price_data_usd(symbols, datetime_interval):
    to_symbol = 'USD'
    exchange = 'CCCAGG'
    data = {}
    for s in symbols:
        data[s] = price_data(s, to_symbol, exchange, datetime_interval)
    response_status = pd.DataFrame(columns=['Response', 'Message'], index=symbols)
    for s in symbols:
        response_status.loc[s, 'Response'] = data[s].get('Response')
        response_status.loc[s, 'Message'] = data[s].get('Message')
    #print('Total amount of coins: %.0f' % len(response_status))
    success_coins = response_status[response_status.Response == 'Success']
    #print('Coins with success: %.0f' % len(success_coins))
    error_coins = response_status[response_status.Response == 'Error']
    #print('Coins with error: %.0f' % len(error_coins))
    return success_coins, error_coins


def download_price_data_usd(symbols, datetime_interval):
    to_symbol = 'USD'
    exchange = 'CCCAGG'
    data = {}
    df = {}
    for s in symbols:
        df[s] = convert_price_data(price_data(s, to_symbol, exchange, datetime_interval))
        df[s].datetime = pd.to_datetime(df[s].datetime)
        df[s].set_index('datetime', inplace=True)
        df[s].sort_index(inplace=True)
        #df[s].rename(columns={'close': s}, inplace=True)
        globals()[datetime_interval + 'data_' + s] = df[s][df[s].close != 0]
        # print('%s data entries %.0f from %s to %s' % (s, len(globals()[datetime_interval + 'data_' + s]),
        #                                               globals()[datetime_interval + 'data_' + s].head(1).index[0],
        #                                               globals()[datetime_interval + 'data_' + s].tail(1).index[0]))
        data[s] = globals()[datetime_interval + 'data_' + s]
    return data


def choose_coins(coins):
    data = download_price_data_usd(check_price_data_usd(cmc_top_coins.symbol.values, 'day')[0].index.values, 'day')
    prices = pd.DataFrame(data['BTC'])
    prices.rename(columns={'close': 'BTC'}, inplace=True)
    for s in check_price_data_usd(cmc_top_coins.symbol.values, 'day')[0].index.values[1:9]:
        prices[s] = pd.DataFrame(data[s]).close
    prices.dropna(inplace=True)
    prices.drop(columns=['USDC', 'BUSD', 'USDT'], inplace=True)
    print('Your have chosen following coins: %s.' % coins)
    coins = coins.split(", ")
    data = prices[coins]
    solana = prices['SOL']
    data = pd.merge(solana, data, on='datetime')
    print('Time horizon: %s to %s.' % (data.head(1).index[0], data.tail(1).index[0]))
    returns = data.pct_change(periods=1)
    returns.dropna(inplace=True)
    return returns


def mvp(coins, limits, solana_investment):
    data = download_price_data_usd(check_price_data_usd(cmc_top_coins.symbol.values, 'day')[0].index.values, 'day')
    prices = pd.DataFrame(data['BTC'])
    prices.rename(columns={'close': 'BTC'}, inplace=True)
    for s in check_price_data_usd(cmc_top_coins.symbol.values, 'day')[0].index.values[1:9]:
        prices[s] = pd.DataFrame(data[s]).close
    prices.dropna(inplace=True)
    prices.drop(columns=['USDC', 'BUSD', 'USDT'], inplace=True)
    print('Your have chosen following coins: %s.' % coins)
    coins = coins.split(", ")
    data = prices[coins]
    solana = prices['SOL']
    data = pd.merge(solana, data, on='datetime')
    print('Time horizon: %s to %s.' % (data.head(1).index[0], data.tail(1).index[0]))
    returns = data.pct_change(periods=1)
    returns.dropna(inplace=True)
    coins = returns.columns.values
    returns.dropna(inplace=True)
    def objective(weights):
        weights = np.array(weights)
        covariance_matrix = returns.cov()
        return weights.dot(covariance_matrix).dot(weights.T)
    lim = limits.split(", ")
    solana_limit = solana_investment
    if len(coins) == 6:
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "ineq", "fun": lambda x: x[0] - float(solana_limit)},
                {"type": "ineq", "fun": lambda x: 0.5 - x[0]},
                {"type": "ineq", "fun": lambda x: x[1] - float(lim[0])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[1]},
                {"type": "ineq", "fun": lambda x: x[2] - float(lim[1])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[2]},
                {"type": "ineq", "fun": lambda x: x[3] - float(lim[2])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[3]},
                {"type": "ineq", "fun": lambda x: x[4] - float(lim[3])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[4]},
                {"type": "ineq", "fun": lambda x: x[5] - float(lim[4])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[5]})
    if len(coins) == 5:
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "ineq", "fun": lambda x: x[0] - float(solana_limit)},
                {"type": "ineq", "fun": lambda x: 0.5 - x[0]},
                {"type": "ineq", "fun": lambda x: x[1] - float(lim[0])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[1]},
                {"type": "ineq", "fun": lambda x: x[2] - float(lim[1])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[2]},
                {"type": "ineq", "fun": lambda x: x[3] - float(lim[2])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[3]},
                {"type": "ineq", "fun": lambda x: x[4] - float(lim[3])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[4]})
    if len(coins) == 4:
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "ineq", "fun": lambda x: x[0] - float(solana_limit)},
                {"type": "ineq", "fun": lambda x: 0.5 - x[0]},
                {"type": "ineq", "fun": lambda x: x[1] - float(lim[0])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[1]},
                {"type": "ineq", "fun": lambda x: x[2] - float(lim[1])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[2]},
                {"type": "ineq", "fun": lambda x: x[3] - float(lim[2])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[3]})
    if len(coins) == 3:
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "ineq", "fun": lambda x: x[0] - float(solana_limit)},
                {"type": "ineq", "fun": lambda x: 0.5 - x[0]},
                {"type": "ineq", "fun": lambda x: x[1] - float(lim[0])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[1]},
                {"type": "ineq", "fun": lambda x: x[2] - float(lim[1])},
                {"type": "ineq", "fun": lambda x: 0.5 - x[2]})
    bounds = tuple((-1, 1) for x in range(returns.shape[1]))
    naive = [1. / returns.shape[1] for x in range(returns.shape[1])]
    optimization = minimize(objective, naive, method="SLSQP", bounds=bounds, constraints=cons)
    mvp_weights = pd.DataFrame({'Symbol': returns.columns, 'Weights': optimization.x})
    mvp_weights.set_index('Symbol', inplace=True)
    pd.options.display.float_format = "{:,.6f}".format
    port_volatility = pow(optimization.fun, 1/2)
    print('CONGRATULATION you created your optimal portfolio: ')
    print('Minimal daily portfolio risk is ' "{:.2%}".format(port_volatility))
    ret_weights = mvp_weights.T * returns.mean()
    port_return = ret_weights.T.sum().values
    print('Daily portfolio return is ' "{:.2%}".format(float(port_return)))
    sharpe = port_return / port_volatility
    print('Portfolio sharpe ratio is %.2f' % float(sharpe))
    print('Assets allocation:')
    for c in coins:
        print(c + ' amount is ' "{:.2%}".format(float(mvp_weights.loc[c, ('Weights')])))
    print("\x1B[3m" + 'Please note that the maximus investment in one could can not exceed 50 %' + "\x1B[0m")
    return port_volatility, port_return


def additional_info(coins, limits, solana_investment, x):
    if x == 'ok':
        print("\x1B[3m" + 'Please wait, i am calculating :)' + "\x1B[0m")
        data = download_price_data_usd(check_price_data_usd(cmc_top_coins.symbol.values, 'day')[0].index.values, 'day')
        prices = pd.DataFrame(data['BTC'])
        prices.rename(columns={'close': 'BTC'}, inplace=True)
        for s in check_price_data_usd(cmc_top_coins.symbol.values, 'day')[0].index.values[1:9]:
            prices[s] = pd.DataFrame(data[s]).close
        prices.dropna(inplace=True)
        prices.drop(columns=['USDC', 'BUSD', 'USDT'], inplace=True)
        coins = coins.split(", ")
        data = prices[coins]
        solana = prices['SOL']
        data = pd.merge(solana, data, on='datetime')
        returns = data.pct_change(periods=1)
        returns.dropna(inplace=True)
        print('Return statistics based on daily data.')
        pd.options.display.float_format = "{:.2%}".format
        print(returns.describe().loc[['mean', 'std', 'min', 'max']])
        corr = returns.corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        heatmap(corr, cmap='viridis', xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt=".2f", linewidth=0.3)
        ax.xaxis.tick_top()
        title = 'Correlation Matrix'.upper()
        plt.title(title, loc='left')
        #plt.savefig('correls.png', dpi='figure', format='png')
        plt.show()
        print("Correlation matrix: The more vibrant the colour is, the stronger the co-movement of coins.")
        print("\x1B[3m" + 'Please wait, i am plotting all possible portfolio combinations :)' + "\x1B[0m")
        num_ports = 10000
        all_weights = np.zeros((num_ports, len(returns.columns)))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)
        for i in range(num_ports):
            weights = np.random.random(size=len(returns.columns))
            weights = weights / np.sum(weights)
            all_weights[i, :] = weights
            ret_arr[i] = np.sum(returns.mean() * weights) * 100
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * 100
            sharpe_arr[i] = ret_arr[i] / vol_arr[i]
        coins = returns.columns.values
        def objective(weights):
            weights = np.array(weights)
            covariance_matrix = returns.cov()
            return weights.dot(covariance_matrix).dot(weights.T)
        lim = limits.split(", ")
        solana_limit = solana_investment
        if len(coins) == 6:
            cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                    {"type": "ineq", "fun": lambda x: x[0] - float(solana_limit)},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[0]},
                    {"type": "ineq", "fun": lambda x: x[1] - float(lim[0])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[1]},
                    {"type": "ineq", "fun": lambda x: x[2] - float(lim[1])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[2]},
                    {"type": "ineq", "fun": lambda x: x[3] - float(lim[2])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[3]},
                    {"type": "ineq", "fun": lambda x: x[4] - float(lim[3])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[4]},
                    {"type": "ineq", "fun": lambda x: x[5] - float(lim[4])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[5]})
        if len(coins) == 5:
            cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                    {"type": "ineq", "fun": lambda x: x[0] - float(solana_limit)},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[0]},
                    {"type": "ineq", "fun": lambda x: x[1] - float(lim[0])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[1]},
                    {"type": "ineq", "fun": lambda x: x[2] - float(lim[1])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[2]},
                    {"type": "ineq", "fun": lambda x: x[3] - float(lim[2])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[3]},
                    {"type": "ineq", "fun": lambda x: x[4] - float(lim[3])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[4]})
        if len(coins) == 4:
            cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                    {"type": "ineq", "fun": lambda x: x[0] - float(solana_limit)},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[0]},
                    {"type": "ineq", "fun": lambda x: x[1] - float(lim[0])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[1]},
                    {"type": "ineq", "fun": lambda x: x[2] - float(lim[1])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[2]},
                    {"type": "ineq", "fun": lambda x: x[3] - float(lim[2])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[3]})
        if len(coins) == 3:
            cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},
                    {"type": "ineq", "fun": lambda x: x[0] - float(solana_limit)},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[0]},
                    {"type": "ineq", "fun": lambda x: x[1] - float(lim[0])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[1]},
                    {"type": "ineq", "fun": lambda x: x[2] - float(lim[1])},
                    {"type": "ineq", "fun": lambda x: 0.5 - x[2]})
        bounds = tuple((-1, 1) for x in range(returns.shape[1]))
        naive = [1. / returns.shape[1] for x in range(returns.shape[1])]
        optimization = minimize(objective, naive, method="SLSQP", bounds=bounds, constraints=cons)
        mvp_weights = pd.DataFrame({'Symbol': returns.columns, 'Weights': optimization.x})
        mvp_weights.set_index('Symbol', inplace=True)
        port_vol = pow(optimization.fun, 1 / 2)
        ret_weights = mvp_weights.T * returns.mean()
        port_ret = ret_weights.T.sum().values
        port_volatility = port_vol * 100
        port_return = port_ret * 100
        plt.figure(figsize=(11, 9))
        plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', linewidths=0.3)
        plt.colorbar().set_label(label='Sharpe Ratio', size=9)
        plt.xlabel('Daily risk (%)', size=9)
        plt.ylabel('Daily return (%)', size=9)
        plt.title('Possible optimal portfolios')
        # plt.savefig('mvp.png', dpi='figure', format='png')
        plt.show()
        print("The warm colour represents portfolios that pay the highest return for each percentage point of risk.")

