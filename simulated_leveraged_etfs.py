"""
creates triple leveraged simulated datasets going back to the earliest dates for indices

currently does this for QQQ and SPY

Before running this, should be in environment with zipline installed
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import trimboth
from sklearn.ensemble import RandomForestRegressor

# location for saving data
DATADIR = 'eod_data/simulated/'


def check_correlations(df1, df2, plot=False):
    """
    checks for correlations between two dataframes;
    both must have a 'daily_pct_chg' column
    this can be used for checking that a 2 or 3x etf is properly correlated with the underlying asset

    also gets the multiplication factor and standard deviation
    """
    # check correlation to make sure it's high
    both = pd.concat([df1['daily_pct_chg'], df2['daily_pct_chg']], axis=1).dropna()
    both.columns = ['regular', 'leveraged']
    if plot:
        corr = both.corr()  # usually around 99.9 or 99.8
        both.plot.scatter(x='regular', y='leveraged')
        plt.title('correlation: ' + str(round(corr.iloc[0, 1], 4)))
        plt.show()

    # look at distribution of TQQQ return multiples
    t = (both['leveraged'] / both['regular']).fillna(0).to_frame()
    t[np.isinf(t[0])] = 0
    # exclude outliers, which throw off the mean
    # removes right and leftmost 5% of quantiles
    new_t = trimboth(t[0], 0.05)
    # t = t[t[0] < t[0].quantile(0.9)]
    # some large outliers
    # t[(t < 6) & (t > -6)].hist(bins=50)

    if plot:
        plt.hist(new_t, bins=50)
        plt.show()

    print('mean and std for multiples:')
    avg, std = new_t.mean(), new_t.std()
    print(avg)
    print(std)
    return avg


def simulate_leveraged(stocks, etf_names=['QQQ', 'TQQQ', 'SQQQ'], return_dfs=False, write=True):
    """
    creates 3x and 2x leveraged ETFS for historical data
    QQQ, SPY, todo: DJI
    if write, writes to /home/nate/Dropbox/data/eod_data/

    stocks is dictionary of dataframes
    etf_names is list of 3 etfs: 1 that is the base (unleveraged) ETF, then 1 that is the positive etf, then one that is the negative leveraged ETF
    if return_dfs is True, will return the simulated dfs
    if write is True, writes simulated dfs to file
    """
    normal = stocks[etf_names[0]].copy()
    pos = stocks[etf_names[1]].copy()
    neg = stocks[etf_names[2]].copy()
    pos_sim, neg_sim = create_simulated_leveraged(normal, pos, neg)

    # change columns for zipline format
    normal.reset_index(inplace=True)
    # Zipline will adjust the prices for splits and dividends I think, but better to just use
    # Quandl-adjusted prices
    normal = normal[['Date', 'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume', 'Dividend', 'Split']]
    normal.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividend', 'split']

    normal['dividend'] = 0
    normal['split'] = 0

    # this version uses the normal prices with splits
    # normal = normal[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividend', 'Split']]
    # normal.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividend', 'split']

    if write:
        # normal['volume'] = normal['volume'].astype('int')
        # pos_sim['volume'] = pos_sim['volume'].astype('int')
        # neg_sim['volume'] = neg_sim['volume'].astype('int')
        normal.to_csv(DATADIR + etf_names[0] + '.csv', index=False)
        pos_sim.to_csv(DATADIR + etf_names[1] + '.csv', index=False)
        neg_sim.to_csv(DATADIR + etf_names[2] + '.csv', index=False)

    if return_dfs:
        return normal, pos_sim, neg_sim


def create_simulated_leveraged(df, pos_lev_df, neg_lev_df):
    # get max and min from untouched pos/neg leveraged dfs for
    # price adjusting later
    pos_max = pos_lev_df['Close'].max()
    pos_min = pos_lev_df['Close'].min()
    neg_max = neg_lev_df['Close'].max()
    neg_min = neg_lev_df['Close'].min()

    # TODO: need to add in splits to keep price between 40 and 180 for TQQQ, in between 10 and 80 for SQQQ
    # usually about 2.97 for a 3x etf
    df['daily_pct_chg'] = df['Adj_Close'].pct_change()
    pos_lev_df['daily_pct_chg'] = pos_lev_df['Adj_Close'].pct_change()
    neg_lev_df['daily_pct_chg'] = neg_lev_df['Adj_Close'].pct_change()
    # get average multipliers for leveraged etfs
    pos_mult = check_correlations(df, pos_lev_df)
    neg_mult = check_correlations(df, neg_lev_df)

    # get earliest dates for leveraged etfs
    earliest_pos = pos_lev_df.index.min()
    earliest_neg = neg_lev_df.index.min()
    # make dataframes to hold simulated data
    pos_sim = df.loc[:earliest_pos].copy()
    neg_sim = df.loc[:earliest_neg].copy()

    # simulate leveraged from beginning of normal ETF timeframe and calculate returns
    col = 'Adj_Close'  # only works with Adj_Close for now
    # also need to set 'Close' column for calculating splits, etc
    pos_lev_df['daily_pct_chg'] = pos_lev_df[col].pct_change()
    neg_lev_df['daily_pct_chg'] = neg_lev_df[col].pct_change()
    pos_sim.loc[:, 'daily_pct_chg'] *= pos_mult  # multiply the original ETF by the average multiple to get leveraged amount
    # backcalculate adjuted close
    pos_sim.loc[pos_sim.index[1]:, col] = ((pos_sim['daily_pct_chg'] + 1).cumprod() * pos_sim.iloc[0][col])[1:]
    neg_sim.loc[:, 'daily_pct_chg'] *= neg_mult
    neg_sim.loc[neg_sim.index[1]:, col] = ((neg_sim['daily_pct_chg'] + 1).cumprod() * neg_sim.iloc[0][col])[1:]
    # adjust to match latest tqqq price
    pos_sim['Close'] = pos_sim['Adj_Close']
    neg_sim['Close'] = neg_sim['Adj_Close']

    ratio = pos_lev_df.iloc[0]['Close'] / pos_sim.iloc[-1]['Close']
    pos_sim['Close'] *= ratio
    ratio = pos_lev_df.iloc[0]['Adj_Close'] / pos_sim.iloc[-1]['Adj_Close']
    pos_sim['Adj_Close'] *= ratio

    # adjust to neg leverage price
    ratio = neg_lev_df.iloc[0]['Close'] / neg_sim.iloc[-1]['Close']
    neg_sim['Close'] *= ratio
    ratio = neg_lev_df.iloc[0]['Adj_Close'] / neg_sim.iloc[-1]['Adj_Close']
    neg_sim['Adj_Close'] *= ratio

    pos_sim['Split'] = 1
    neg_sim['Split'] = 1
    pos_sim['Dividend'] = 0
    neg_sim['Dividend'] = 0

    # contain prices between 40 and 180 for positive leverage, 10 and 80 for negative
    low_adj = 0.25  # noticed from SQQQ
    high_adj = 2  # noticed from TQQQ
    # go backwards thru dataset, since the latest price matches the latest price in the real data
    # also start at the 2nd-to-last point, because the last one overlaps the actual data and can't be adjusted
    # problem with getting infinity for earliest very large values...just set splits, then do a calculation based on the compound splits
    total_split = 1
    pos_splits = []
    for i, r in pos_sim.iloc[-1::-1].iterrows():
        # adjust to total split adjustment to this point
        r['Close'] /= total_split
        pos_sim.loc[i, 'Close'] = r['Close']
        if r['Close'] < pos_min:
            # print('less')
            r['Split'] = low_adj
            pos_splits.append(low_adj)
        elif r['Close'] > pos_max:
            # print('more')
            r['Split'] = high_adj
            pos_splits.append(high_adj)
        else:
            pos_splits.append(1)

        total_split *= r['Split']


    # TODO: fix issue where first price is not within range
    low_adj = 0.25  # noticed from SQQQ
    high_adj = 2  # noticed from TQQQ
    total_split = 1
    neg_splits = []
    # doesn't work with .iloc[1:-1:-1]
    for i, r in neg_sim.iloc[-1::-1].iterrows():
        r['Close'] /= total_split
        neg_sim.loc[i, 'Close'] = r['Close']
        if r['Close'] < neg_min:
            r['Split'] = low_adj
            neg_splits.append(low_adj)
        elif r['Close'] > neg_max:
            r['Split'] = high_adj
            neg_splits.append(high_adj)
        else:
            neg_splits.append(1)

        total_split *= r['Split']


    pos_sim['Ticker'] = pos_lev_df['Ticker'][0]
    neg_sim['Ticker'] = neg_lev_df['Ticker'][0]

    pos_sim['Split'] = pos_splits
    neg_sim['Split'] = neg_splits
    pos_sim['Dividend'] = 0
    neg_sim['Dividend'] = 0

    pos_sim_full = pd.concat([pos_sim.iloc[:-1], pos_lev_df])
    neg_sim_full = pd.concat([neg_sim.iloc[:-1], neg_lev_df])

    pos_sim_full.reset_index(inplace=True)
    neg_sim_full.reset_index(inplace=True)

    # rename columns for zipline import
    # renaming adjusted columns as plain close, etc

    # need to use non-adjusted column, otherwise values are too big for
    # negative leveraged (SQQQ as of 2-19-2019)
    # old way, using adjusted columns as normal:
    new_columns = ['date', 'Ticker', 'open', 'high', 'low', 'close', 'volume', 'dividend', 'split',
                   'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Volume', 'daily_pct_chg']
    pos_sim_full.columns = new_columns
    neg_sim_full.columns = new_columns
    # can't have other columns like the 'Ticker' column in there or throws an error
    # https://www.zipline.io/bundles.html#ingesting-data-from-csv-files

    pos_sim_full = pos_sim_full[['date', 'open', 'high', 'low', 'close', 'volume', 'dividend', 'split']]
    neg_sim_full = neg_sim_full[['date', 'open', 'high', 'low', 'close', 'volume', 'dividend', 'split']]

    # with zipline, it seems like if the adj_close price is used (necessary for indicators like MA)
    # then you have to have all 'split' be 1, and dividend be 0

    # TODO: check that dividends remain proper with added splits

    # get predicted OHL (open, high, low)
    pos_sim_full, neg_sim_full = predict_OHL(df, pos_sim_full, neg_sim_full, earliest_pos, earliest_neg)

    return pos_sim_full, neg_sim_full


def predict_OHL(normal, pos_sim, neg_sim, earliest_pos_date, earliest_neg_date):
    """
    Takes 3 dataframes and uses ML to predict open, high, low (OHL) of leveraged ETFs from an unleveraged ETF.
    args:
    normal -- unleveraged ETF pandas dataframe
    pos_sim -- simulated positive leverage ETF pandas dataframe
    neg_sim -- simulated negative leverage ETF pandas dataframe
    earliest_pos_date -- datetime, earliest date with non-simulated positive leveraged ETF data
    earliest_pos_date -- datetime, earliest date with non-simulated negative leveraged ETF data
    """
    # create features and targets
    feat_targ_df = normal.copy()
    pos_sim.set_index('date', inplace=True)
    neg_sim.set_index('date', inplace=True)
    # lowercase to match simulated leveraged data columns
    feat_targ_df.columns = [c.lower() for c in feat_targ_df.columns]
    # actually already calculated in create_simulated_leveraged() as 'daily_pct_change'
    feat_targ_df['adj_close_pct_normal'] = feat_targ_df['adj_close'].pct_change()
    feat_cols = ['adj_close_pct_normal']
    # earliest start date for pos and neg may be different, so treat them separately
    pos_targ_cols = []
    neg_targ_cols = []
    # need to add a label to differentiate them in the combined DF
    for label, d in zip(['normal', 'pos', 'neg'], [feat_targ_df, pos_sim, neg_sim]):
        for bar in ['open', 'high', 'low']:
            # a little confusing, but the 'norm' in there is for normalized, since we are dividing by the Adj_Close
            # 'normal' means the non-leveraged ETF
            feat_targ_df['{}_norm_{}'.format(bar, label)] = d['{}'.format(bar)] / d['close']
            if label == 'pos':
                pos_targ_cols.append('{}_norm_{}'.format(bar, label))
            elif label == 'neg':
                neg_targ_cols.append('{}_norm_{}'.format(bar, label))
            else:
                feat_cols.append('{}_norm_{}'.format(bar, label))

    # first point has an NA since pct_change can't be calculated
    feat_targ_df.dropna(inplace=True)
    # get targets from known data
    train_feats_pos = feat_targ_df[feat_cols].loc[earliest_pos_date:]
    train_feats_neg = feat_targ_df[feat_cols].loc[earliest_neg_date:]
    train_targs_pos = feat_targ_df[pos_targ_cols].loc[earliest_pos_date:]
    train_targs_neg = feat_targ_df[neg_targ_cols].loc[earliest_neg_date:]
    # get feats for unknown data
    test_feats_pos = feat_targ_df[feat_cols].loc[:earliest_pos_date]
    test_feats_neg = feat_targ_df[feat_cols].loc[:earliest_neg_date]

    # predict OHL here
    # hyperparameters have not been tuned, but set to 'default' decent values
    rfr = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=-1, random_state=42, oob_score=True)
    rfr.fit(train_feats_pos, train_targs_pos)
    train_score = rfr.oob_score_
    if train_score < 0.8:
        print('WARNING: OOB score on train data was low:', train_score)

    pos_test_preds = pd.DataFrame(rfr.predict(test_feats_pos),
                                  columns=['open', 'high', 'low'],
                                  index=feat_targ_df.loc[:earliest_pos_date].index)

    # same thing for negative leveraged ETF
    rfr.fit(train_feats_neg, train_targs_neg)
    train_score = rfr.oob_score_
    if train_score < 0.8:
        print('WARNING: OOB score on train data was low:', train_score)

    neg_test_preds = pd.DataFrame(rfr.predict(test_feats_neg),
                                 columns=['open', 'high', 'low'],
                                 index=feat_targ_df.loc[:earliest_neg_date].index)


    # back-calculate OHL and overwrite original DF
    # first datapoint was dropped due to no pct_change for unleveraged etf
    pos_sim_test = pos_sim.loc[:earliest_pos_date]
    pos_sim_test = pos_sim_test.iloc[1:]
    pos_test_preds = pos_test_preds.multiply(pos_sim_test['close'], axis=0)

    pos_sim = pos_sim.iloc[1:]
    pos_sim.loc[:earliest_pos_date, ['open', 'high', 'low']] = pos_test_preds


    neg_sim_test = neg_sim.loc[:earliest_neg_date]
    neg_sim_test = neg_sim_test.iloc[1:]
    neg_test_preds = neg_test_preds.multiply(neg_sim_test['close'], axis=0)

    neg_sim = neg_sim.iloc[1:]
    neg_sim.loc[:earliest_neg_date, ['open', 'high', 'low']] = neg_test_preds

    # need to make 'date' index a column again
    pos_sim.reset_index(inplace=True)
    neg_sim.reset_index(inplace=True)

    return pos_sim, neg_sim


def create_simulated_dataset(stocks, etf_names=['QQQ', 'TQQQ', 'SQQQ'], return_dfs=False, clean_all_but_latest=True, create_dir=True):
    """
    creates simulated dataset for quandl

    args:
    stocks -- dictionary of stocks' dataframes from stock_prediction/download_quandl_EOD.py
    etf_names -- list of tickers (strings); should be normal index, then positive
                leveraged index, then negative leveraged index
    return_dfs -- Boolean; if True, will return normal and simulated dataframes from
                function
    clean_all_but_latest -- Boolean; if True, will clean up all zipline data folders for csv except the latest one
    """
    returned = simulate_leveraged(stocks, etf_names=etf_names, return_dfs=return_dfs, write=True)

    if create_dir:
        create_csvdir_dataset(clean_all_but_latest=clean_all_but_latest)

    if return_dfs:
        return returned


def create_csvdir_dataset(clean_all_but_latest=True):
    """
    creates csv dataset for zipline backtesting
    """
    os.system('CSVDIR=/home/nate/Dropbox/data/eod_data/simulated/ python3 -m zipline ingest -b csvdir')
    if clean_all_but_latest:
        os.system('python3 -m zipline clean -b csvdir --keep-last 1')


def load_sample_data():
    """
    loads QQQ, TQQQ, SQQQ sample data from eod_data folder
    """
    qqq = pd.read_csv('eod_data/QQQ.csv', index_col='Date')
    tqqq = pd.read_csv('eod_data/TQQQ.csv', index_col='Date')
    sqqq = pd.read_csv('eod_data/SQQQ.csv', index_col='Date')
    stocks = {'QQQ': qqq, 'TQQQ': tqqq, 'SQQQ': sqqq}
    return stocks

if __name__ == "__main__":
    stocks = load_sample_data()
    dfs = create_simulated_dataset(stocks, ['QQQ', 'TQQQ', 'SQQQ'], create_dir=True, return_dfs=True)
