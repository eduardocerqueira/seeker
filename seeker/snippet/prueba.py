#date: 2022-12-19T16:31:55Z
#url: https://api.github.com/gists/cf16412e111012295b3fb6973bcb6c0c
#owner: https://api.github.com/users/GeraCollante

import copy
from collections import Counter, defaultdict
from math import ceil
import numpy as np
import pandas as pd
from copy import deepcopy
# from score_predictor.data.misc_tools.helpers import *
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm_notebook as tqdm
from unidecode import unidecode
import itertools
from scipy.stats import kurtosis


from score_predictor.data.misc_tools.helpers import *
from score_predictor.data.vocabularies.mappers import *

    
def process_rare_values(X, metadata):
    
    X_processed = X.copy()
    
    # Assure that the DataFrame has the same columns as the metadata
    metadata = {item: metadata[item] for item in set(X.columns) & set(metadata.keys())}
    
    for key, v in metadata.items():
        
        if type(key) == tuple:
            continue
            
        X_processed[key] = X_processed[key].map(lambda x: x if x in v['top'] else 'RARE' if x in v['rare'] else 'OOV')
    
    return X_processed

################################################
########### TRANSACTION FEATURES ###############
################################################


numerical_cols = [
    'amount', 
    'balance'
    ]

cat_cols = [
    'status_transactions', 
    'category', 
    #'name', 
    #'account_type', 
    'subtype', 
    #'marketing_name'
    ]

date_cat_cols = [
    'day_of_week', 
    # 'day_of_month', 
    'week_of_month', 
    'week_of_year', 
    # 'days_lag',
    # 'weeks_lag',
    # 'months_lag', 
    # 'quarter_lag',
    # 'day_of_year',
    'month_of_year', 
    'quarter_of_year',
    'year', 
    'year_q', 
    'year_month', 
    'year_week',
    'is_weekend',
    'last_day', 
    'last_2_days',
    'last_3_days',
    'last_7_days', 
    'last_30_days', 
    'last_60to31_days',
    'last_90to61_days', 
    'last_120to91_days', 
    'last_150to121_days', 
    'last_180to151_days'
    ]


all_cols = cat_cols + date_cat_cols

cat_subset = [
    'status_transactions', 
    'category',
    'subtype'
              ]

metrics = [
    'count', 
    'count_positive',
    'count_negative',
    'sum', 
    'sum_positive', 
    'sum_negative',
    'mean', 
    'mean_positive', 
    'mean_negative',
    'var', 
    'var_positive',
    'var_negative',
    'pct_positive', 
    'pct_negative',
    'max',
    'max_positive',
    'max_negative',
    'min',
    'min_positive',
    'min_negative'
            ]


metrics_subset = [
    'count', 
    'count_positive',
    'count_negative',
    'sum', 
    'sum_positive', 
    'sum_negative',
    'mean', 
    'mean_positive', 
    'mean_negative',
    'var', 
    'var_positive',
    'var_negative',
    'max',
    'max_positive',
    'max_negative',
    'min',
    'min_positive',
    'min_negative'
            ]



def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    if pd.notnull(dt):
    
        first_day = dt.replace(day=1)

        dom = dt.day
        adjusted_dom = dom + first_day.weekday()

        return int(ceil(adjusted_dom/7.0))

def process_transactions_data(transactions_df):
    
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    transactions_df['amount'] = transactions_df['amount'].astype(float)
    transactions_df['balance'] = transactions_df['balance'].astype(float)
    
    # Create dates attributes
    transactions_df['year'] = transactions_df['date'].dt.year
    transactions_df['month_of_year'] = transactions_df['date'].dt.month
    transactions_df['day_of_month'] = transactions_df['date'].dt.day
    transactions_df['day_of_year'] = transactions_df['date'].dt.dayofyear
    transactions_df['day_of_week'] = transactions_df['date'].dt.weekday
    transactions_df['is_weekend'] = transactions_df['day_of_week'].apply(lambda x: x >= 5)
    transactions_df['week_of_year'] = transactions_df['date'].dt.weekofyear
    transactions_df['quarter_of_year'] = transactions_df['date'].dt.quarter
    transactions_df['week_of_month'] = transactions_df['date'].apply(week_of_month)
    
    
    # Create dates ranks, intervals with respect to created_at and time deltas between transactions

    # rank of transaction for each item_id
    transactions_df['rank'] = transactions_df.groupby('id', sort=True)['date'].rank(method='first', ascending=True).astype(int)

    # # months between transaction date and loan creation date
    # transactions_df['quarter_lag'] = transactions_df.apply(lambda x: (x['loan_created_at'].year - x['date'].year) * 4 \
    #                                                              + x['loan_created_at'].quarter - x['date'].quarter, 1)

    # # months between transaction date and loan creation date
    # transactions_df['months_lag'] = transactions_df.apply(lambda x: (x['loan_created_at'].year - x['date'].year) * 12 \
    #                                                              + x['loan_created_at'].month - x['date'].month, 1)

    # # weeks since creation of loan
    # transactions_df['weeks_lag'] = transactions_df.apply(lambda x: (x['loan_created_at'].year - x['date'].year) * 52 \
    #                                                              + x['loan_created_at'].weekofyear - x['date'].weekofyear, 1)

    # days since creation of loan
    transactions_df['days_lag'] = (transactions_df['loan_created_at'].dt.date - transactions_df['date'].dt.date).dt.days

    # days between transactions
    # transactions_df['days_delta'] = transactions_df.groupby('id')['days_lag'].shift(1).fillna(transactions_df['days_lag']).astype(int) - transactions_df['days_lag']
    transactions_df['days_delta'] = transactions_df['days_lag']- transactions_df.groupby('id')['days_lag'].shift(1).fillna(transactions_df['days_lag']).astype(int)


    # Create period dates to use as time series
    transactions_df['year_q'] =  pd.to_datetime(transactions_df['date'],format='%Y-%m').dt.to_period('Q')
    transactions_df['year_month'] =  pd.to_datetime(transactions_df['date'],format='%Y-%m').dt.to_period('M')
    transactions_df['year_week']  = pd.to_datetime(transactions_df['date'],format='%Y-%m').dt.to_period('W')
    transactions_df['year_day']  = pd.to_datetime(transactions_df['date'],format='%Y-%m').dt.to_period('D')
    
    # Create day window based features
    transactions_df['last_3_days'] = np.where(transactions_df['days_lag']<3, 1, 0)
    transactions_df['last_2_days'] = np.where(transactions_df['days_lag']<2, 1, 0)
    transactions_df['last_day'] = np.where(transactions_df['days_lag']<1, 1, 0)
    transactions_df['last_7_days'] = np.where(transactions_df['days_lag']<7, 1, 0)
    transactions_df['last_30_days'] = np.where(transactions_df['days_lag']<30, 1, 0)
    transactions_df['last_60to31_days'] = np.where((transactions_df['days_lag']>29)&(transactions_df['days_lag']<60), 1, 0)
    transactions_df['last_90to61_days'] = np.where((transactions_df['days_lag']>59)&(transactions_df['days_lag']<90), 1, 0)
    transactions_df['last_120to91_days'] = np.where((transactions_df['days_lag']>89)&(transactions_df['days_lag']<120), 1, 0)
    transactions_df['last_150to121_days'] = np.where((transactions_df['days_lag']>119)&(transactions_df['days_lag']<150), 1, 0)
    transactions_df['last_180to151_days'] = np.where((transactions_df['days_lag']>149)&(transactions_df['days_lag']<180), 1, 0)

    # Change sign for credit card transactions
    transactions_df.loc[transactions_df.subtype=='CREDIT_CARD'].amount = transactions_df.loc[transactions_df.subtype=='CREDIT_CARD'].amount*-1

    # Cast Booleans to 0/1
    for col in transactions_df.columns:
        if transactions_df[col].dtype==bool:
            transactions_df[col] = transactions_df[col]*1
            
    return transactions_df


def create_period_average_features(transactions_df_):
    
    transactions_df = transactions_df_.copy()
    
     # Period average metrics
    average_monthly_amount = transactions_df.groupby(
                                        ['id', 
                                        pd.Grouper(key='date', freq='M')]
                )['amount'].sum().unstack(1).fillna(0).agg(['mean'], 1).rename(columns={'mean':''})\
                    .add_suffix('monthly_average')

    average_weekly_amount = transactions_df.groupby(
                                            ['id', 
                                            pd.Grouper(key='date', freq='W')]
                    )['amount'].sum().unstack(1).fillna(0).agg(['mean'], 1).rename(columns={'mean':''})\
                    .add_suffix('weekly_average')

    average_daily_amount = transactions_df.groupby(
                                            ['id', 
                                            pd.Grouper(key='date', freq='D')]
                    )['amount'].sum().unstack(1).fillna(0).agg(['mean'], 1).rename(columns={'mean':''})\
                    .add_suffix('daily_average')
                    
    
    # Period average grouped by categories metrics

    average_monthly_amount_by_category = transactions_df.groupby(
                                                     ['id', 
                                                     pd.Grouper(key='date', freq='M'), 
                                                     'category']
                            )['amount'].sum().unstack(1).fillna(0).mean(1).unstack().fillna(0)\
                            .add_suffix('_monthly_average').add_prefix('category_')

    average_weekly_amount_by_category = transactions_df.groupby(
                                                         ['id', 
                                                         pd.Grouper(key='date', freq='W'), 
                                                         'category']
                                )['amount'].sum().unstack(1).fillna(0).mean(1).unstack().fillna(0)\
                                .add_suffix('_weekly_average').add_prefix('category_')



    average_daily_amount_by_category = transactions_df.groupby(
                                                         ['id', 
                                                         pd.Grouper(key='date', freq='D'), 
                                                         'category']
                                )['amount'].sum().unstack(1).fillna(0).mean(1).unstack().fillna(0)\
                                .add_suffix('_daily_average').add_prefix('category_')

    
    transactions_feats_df = average_monthly_amount\
                            .join(average_weekly_amount)\
                            .join(average_daily_amount)\
                            .join(average_monthly_amount_by_category)\
                            .join(average_weekly_amount_by_category)\
                            .join(average_daily_amount_by_category)
    
    return transactions_feats_df
    
    
def create_grouped_features(transactions_df_):
    
    transactions_df = transactions_df_.copy()
    
    feats_df_dict = {}

    for col in all_cols: 
        ts_df = transactions_df.groupby(['id']+[col]).apply(compute_metrics)
        
        ts_feats_df = pd.pivot_table(ts_df.reset_index(), 
                        values = ts_df.columns, 
                        index = ['id'], 
                        columns = col)
        ts_feats_df.columns = ts_feats_df.columns.to_flat_index()
        ts_feats_df.columns = ['_'.join(str(v) for v in col) for col in ts_feats_df.columns.values]
        
        feats_df_dict[col] = ts_feats_df
    
    # feats_df_comb = pd.concat(feats_df_comb_dict, axis=1)
    feats_df = pd.concat(feats_df_dict, axis=1)

    feats_df.columns = feats_df.columns.to_flat_index()
    feats_df.columns = ['_'.join(str(v) for v in col) for col in feats_df.columns.values]
    
    return feats_df

def create_combinatorial_grouped_features(transactions_df_):
    
    transactions_df = transactions_df_.copy()
            
    feats_df_comb_dict = {}
    
    transactions_df_pos = transactions_df[transactions_df.amount>=0]
    transactions_df_neg = transactions_df[transactions_df.amount<0]

    list_of_cat_date_combinations = create_col_combinations(all_cols)

    for comb in tqdm(list_of_cat_date_combinations):
        ts_df = transactions_df.groupby(['id']+comb)
        ts_df_pos = transactions_df_pos.groupby(['id']+comb)
        ts_df_neg = transactions_df_neg.groupby(['id']+comb)

        features_dataframes = {}
        features_dataframes['count'] = ts_df['amount'].count()
        features_dataframes['count_positive'] = ts_df_pos['amount'].count()
        features_dataframes['count_negative'] = ts_df_neg['amount'].count()

        features_dataframes['sum'] = ts_df['amount'].sum()
        features_dataframes['sum_positive'] = ts_df_pos['amount'].sum()
        features_dataframes['sum_negative'] = ts_df_neg['amount'].sum()

        features_dataframes['mean'] = ts_df['amount'].mean()
        features_dataframes['mean_positive'] = ts_df_pos['amount'].mean()
        features_dataframes['mean_negative'] = ts_df_neg['amount'].mean()

        features_dataframes['var'] = ts_df['amount'].var()
        features_dataframes['var_positive'] = ts_df_pos['amount'].var()
        features_dataframes['var_negative'] = ts_df_neg['amount'].var()
        
        features_dataframes['max'] = ts_df['amount'].max()
        features_dataframes['max_positive'] = ts_df_pos['amount'].max()
        features_dataframes['max_negative'] = ts_df_neg['amount'].max()
        
        features_dataframes['min'] = ts_df['amount'].min()
        features_dataframes['min_positive'] = ts_df_pos['amount'].min()
        features_dataframes['min_negative'] = ts_df_neg['amount'].min()
        
        all_features_df = pd.DataFrame(features_dataframes)
        
        ts_feats_df = pd.pivot_table(all_features_df.reset_index(), 
                        values = all_features_df.columns, 
                        index = ['id'], 
                        columns = comb)
        ts_feats_df.columns = ts_feats_df.columns.to_flat_index()
        ts_feats_df.columns = ['_'.join(str(v) for v in col) for col in ts_feats_df.columns.values]
        
        feats_df_comb_dict['_'.join(comb)] = ts_feats_df

    feats_df_comb = pd.concat(feats_df_comb_dict, axis=1)

    feats_df_comb.columns = feats_df_comb.columns.to_flat_index()
    feats_df_comb.columns = ['_'.join(str(v) for v in col) for col in feats_df_comb.columns.values]

    return feats_df_comb


def create_transactions_features(transactions_df_, metadata):
    
    transactions_df = transactions_df_.copy()
    transactions_processed_df = process_rare_values(transactions_df, metadata)
    
    grouped_features = create_grouped_features(transactions_processed_df)
    #print('grouped_features')
    combinatorial_grouped_features = create_combinatorial_grouped_features(transactions_df)
    #print('combinatorial_grouped_features')
    period_average_features = create_period_average_features(transactions_processed_df)
    #print('period_average_features')
    
    transactions_feats_df = grouped_features\
                            .join(combinatorial_grouped_features)\
                            .join(period_average_features)\
                            
    # TODO: CONFIRM THIS JOIN IS OK
    transactions_feats_df = transactions_feats_df.join(transactions_df[['id', 'client_user_id']].drop_duplicates(subset=['client_user_id'], keep='first').set_index('id'))
    transactions_feats_df = transactions_feats_df.set_index('client_user_id')

    return transactions_feats_df


def build_transactions_df(df):

    df = df.explode('accounts').reset_index(drop=True)
    
    # for col in ['connector', 'identity', 'accounts']:
    #     df = df.merge(df[col].dropna().apply(pd.Series), 
    #                   suffixes=['', f'_{col}'], left_index=True, right_index=True)
    #     del df[col]
    
    df = df.merge(df['accounts'].dropna().apply(pd.Series), 
                    suffixes=['', '_accounts'], left_index=True, right_index=True)
    del df['accounts']

    df = df.explode('transactions').reset_index(drop=True)

    df = df.merge(df['transactions'].dropna().apply(pd.Series), 
                  suffixes=['', '_transactions'], left_index=True, right_index=True)

    del df['transactions']

    return df

def compute_metrics(x):
    d = {}
    for col in numerical_cols:
        
        d[f'{col}_count'] = x[col].count()
        d[f'{col}_count_positive'] = sum(x[col] >= 0)
        d[f'{col}_count_negative'] = sum(x[col] < 0)
        
        d[f'{col}_sum'] = x[col].sum()
        d[f'{col}_sum_positive'] = x[x[col] >= 0][col].sum()
        d[f'{col}_sum_negative'] = x[x[col] < 0][col].sum()
        
        d[f'{col}_mean'] = x[col].mean()
        d[f'{col}_mean_positive'] = x[x[col] >= 0][col].mean()
        d[f'{col}_mean_negative'] = x[x[col] < 0][col].mean()
        
        d[f'{col}_var'] = x[col].var()
        d[f'{col}_var_positive'] = x[x[col] >= 0][col].var()
        d[f'{col}_var_negative'] = x[x[col] < 0][col].var()
        
        d[f'{col}_pct_positive'] = np.mean(x[col] >= 0)
        d[f'{col}_pct_negative'] = np.mean(x[col] < 0)
        
        d[f'{col}_max'] = x[col].max()
        d[f'{col}_max_positive'] = x[x[col] >= 0][col].max()
        d[f'{col}_max_negative'] = x[x[col] < 0][col].max()
        
        d[f'{col}_min'] = x[col].min()
        d[f'{col}_min_positive'] = x[x[col] >= 0][col].min()
        d[f'{col}_mean_negative'] = x[x[col] < 0][col].min()
        
            
    return pd.Series(d)

def create_col_combinations(cat_cols_list):
    
    # Create all combinations of categorical columns for grouping
    cat_subset_ = [col for col in cat_subset if col in cat_cols_list]
    date_cat_cols_ = [col for col in date_cat_cols if col in cat_cols_list]
    
    list_of_combinations = []
    for i in range(1, len(cat_subset_)):
        list_of_combinations.append(list(itertools.combinations(cat_subset_, i)))
    list_of_combinations = [j for i in list_of_combinations for j in i]
    list_of_combinations = [list(tup) for tup in list_of_combinations]

    # Create combinations of categorical + date features for grouping
    list_of_cat_date_combinations = []
    for cat in list_of_combinations:
        for date_cat in date_cat_cols_:
            list_of_cat_date_combinations.append(cat+[date_cat])
    
    return list_of_cat_date_combinations


################################################
############# ACCOUNT FEATURES #################
################################################

account_cat_cols = [
    "status", "execution_status", 
    "connector.country", "connector.name", 
    "connector.type", "income.status"
    ]

account_other_cols = [
    "accounts",
    "connector.products",
    "identity.phone_numbers",
    "identity.emails",
    "income.irregular_income_source.average_monthly_income_last_30_days",
    "income.irregular_income_source.average_monthly_income_last_90_days",
    "income.irregular_income_source.average_monthly_income_last_180_days",
    "income.irregular_income_source.average_monthly_income_last_360_days",
    "income.irregular_income_source.days_covered_with_income",
    "income.irregular_income_source.num_income_transactions",
    "income.average_monthly_income",
    "income.consistency",
    "income.longevity",
    "income.regularity",
    "income.average_monthly_income_last_30_days",
    "income.average_monthly_income_last_90_days",
    "income.average_monthly_income_last_180_days",
    "income.average_monthly_income_last_360_days",
    "income.days_covered_with_income",
    "income.num_income_transactions",
]

accounts_cols = account_cat_cols + account_other_cols


ac_card_providers = {'mc':'mastercard', 
                 'vs':'visa', 
                 'elo':'elo', 
                 'ourocard':'ourocard', 
                 'hipercard':'hipercard', 
                 'itau':'itau', 
                 }

ac_account_type = ['conta corrente', 'conta remunerada']

ac_card_level = {'signature':3, 'platinum':2, 'gold':1, 'black':3}

account_types = ['BANK', 'CREDIT']

account_subtypes = ['CHECKING_ACCOUNT', 'SAVINGS_ACCOUNT', 'CREDIT_CARD']

execution_statuses = ['SUCCESS', 'PARTIAL_SUCCESS']

bank_names = ['Nubank', 'Santander', 'Bradesco', 'Banco do Brasil', 'Itaú', 'Banco Inter']

mn_providers = [
     'AADVANTAGE',
     'CAIXA',
     'DECOLAR',
     'DOTZ',
     'NEO',
     'OUROCARD',
     'PRIVATE',
     'SANTANDER',
     'SMILES'
            ]

mn_levels = [
    'BASICO',
    'ELITE',
    'FIT',
    'GOLD',
    'INTERNACIONAL',
    'NACIONAL',
    'SX',
    'UNIQUE',
    'PLATINUM'
]

mn_card_companies = [
    'VISA', 'ELO', 'MASTER'
]

numeric_ops = {
    "sum": np.sum,
    "mean": np.mean,
    "std": np.std,
    "max": np.max,
    "min": np.min,
    "kurt": kurtosis
}

cd_brands = ["ELO", "MASTERCARD", "VISA"]
cd_levels = {'GOLD':1, 'PLATINUM':2, 'SIGNATURE':3, 'INFINITE':4}

bank_products = ['ACCOUNTS',
 'CREDIT_CARDS',
 'IDENTITY',
 'INVESTMENTS',
 'INVESTMENTS_TRANSACTIONS',
 'PAYMENT_DATA',
 'TRANSACTIONS']

phone_types = ["Personal", "Work"]
email_types = ["Personal", "Work"]

account_fields=[
#     'account_id', 
    'account_type', 
    'subtype', 
#     'number', 
    'name', 
    'marketing_name', 
    'balance', 
#     'tax_number', 
    'owner', 
#     'currency_code', 
    'bank_data', 
    'credit_data'
]


def process_accounts(x):
    """
    Process row "accounts" from the json of the user.
    Parameters
    ----------
    x : dict
        List with the accounts
    Returns
    -------
    dict
        Features created
    """
    my_lists = {}
    features = {}
    
    for field in account_fields:
        if x:
            my_lists[field] = [item.get(field) for item in x if item]
        else:
            my_lists[field] = []
            
    account_type_features = account_type(my_lists["account_type"])
    subtype_features = account_subtype(my_lists["subtype"])
    account_name_features = account_name(my_lists["name"])
    marketing_name_features = marketing_name(my_lists["marketing_name"])
    account_balance_features = account_balance(my_lists["balance"])
    # owner_features = owner(my_lists["owner"])
    bank_data_features = bank_data(my_lists["bank_data"])
    credit_data_features = credit_data(my_lists["credit_data"])
    
    features = {**account_type_features, **subtype_features, 
                **account_name_features, **marketing_name_features, 
                **account_balance_features, # **owner_features,
                **bank_data_features, **credit_data_features}
    
    return pd.Series({'accounts.' + k:v for k, v in features.items()})


def process_connector(x):
    """
    Process row "connector" (bank) from the json of the user.
    Parameters
    ----------
    x : dict
        List with the bank names and the products of each bank
    Returns
    -------
    dict
        Features created
    """
    features = {}
    bank_name = x.get("name")
    products = x.get("products")
    
    bank_names_features = dict.fromkeys(bank_names, 0)
    bank_products_features = dict.fromkeys(bank_products, 0)
    
    if bank_name:
        bank_names_features[bank_name] = 1
        
    for product in products:
        bank_products_features[product] = 1
    
    bank_names_features = {"bank_name_" + k:v for k,v in bank_names_features.items()}
    bank_products_features = {"bank_products_" + k:v for k,v in bank_products_features.items()}
    
    features = {**bank_names_features, **bank_products_features}
    return {'connector.' + k:v for k,v in features.items()}


def process_identity(x):
    """
    Process row "identity" from the json of the user.
    Parameters
    ----------
    x : dict
        Dict with phone numbers and emails of the user
    Returns
    -------
    dict
        Features created
    """
    features = {}
    
    if x and x.get("phone_numbers"):
        phone_numbers_type = [item.get("type") for item in x["phone_numbers"] if item]
        phone_numbers = [item.get("value") for item in x["phone_numbers"] if item]
    else:
        phone_numbers_type = []
        phone_numbers = []

    if x and x.get("emails"):
        emails_type = [item["type"] for item in x["emails"] if item]
        emails = [item["value"] for item in x["emails"] if item]
    else:
        emails_type = []
        emails = []

    for phone_type in phone_types:
        features["phone_" + phone_type] = phone_numbers_type.count(phone_type)
    for email_type in email_types:
        features["email_" + email_type] = emails_type.count(email_type)
    
    features["phone_numbers"] = phone_numbers
    features["emails"] = emails
    
    return {"identity." + k:v for k,v in features.items()}


def account_balance(my_list):
    features = {}
    balances = np.array([item for item in my_list if item]).astype(float)
    
    for op, func in numeric_ops.items():
        if len(balances) > 0:
            features[op] = func(balances)
        else:
            features[op] = np.nan
            
    return {'account_balance.' + k:v for k,v in features.items()}


def account_name(my_list):
    """
    Create features from a list of account_names.
    It uses ac_account_type, ac_card_providers 
    and ac_card_level to create the features.
    Parameters
    ----------
    my_list : list
        List of account_names
    Returns
    -------
    dict
        Features created
    """
    features = {}
    my_list = [item.lower() for item in my_list if item]
    
    # Account type
    for account in ac_account_type:
        features['account_type_' + account] = any([account in item for item in my_list])
        
    # Card providers
    for card_key, card_value in ac_card_providers.items():
        features['card_provider_' + card_key] = any([card_value in item for item in my_list])
        
    # Card levels
    features['card_level_sum'] = 0
    for card_key, card_value in ac_card_level.items():
        for item in my_list:
            if card_key in item:
                features['card_level_sum'] += card_value
                features['card_level_' + card_key] = True
            else:
                features['card_level_' + card_key] = False
        
    return {'account_name.' + k:v for k,v in features.items()}


def account_subtype(my_list):

    """
    Create features from a list of account_subtypes.
    Parameters
    ----------
    my_list : list
        List of account_subtypes
    Returns
    -------
    dict
        Features created
    """
    features = {}
    
    for account_subtype in account_subtypes:
        features[account_subtype] = my_list.count(account_subtype)
    return {'account_subtype.' + k:v for k,v in features.items()}


def account_type(my_list):
    """
    Create features from a list of account_types.
    It uses ac_account_type to create the features.
    Parameters
    ----------
    my_list : list
        List of account_types
    Returns
    -------
    dict
        Features created
    """
    features = {}
    
    for ac_account_type in account_types:
        features[ac_account_type] = my_list.count(ac_account_type)
    return {'account_type.' + k:v for k,v in features.items()}


def bank_data(bank_data):
    """
    Create features from a list of bank_data.
    Apply numeric operations to closingBalance 
    like sum, mean, std, min, max, etc.
    Parametersahgasen
    ----------
    bank_data : list
        List of bank_data with two keys: 'closingBalance' and 'accountNumber'
        We focus on the closingBalance.
    Returns
    -------
    dict
        Features created
    """
    features = {}
    balances = [item.get('closingBalance') for item in bank_data if item]
    balances = [item for item in balances if item] # cleaning None values

    for op, func in numeric_ops.items():
        if len(balances) > 0:
            features['closingBalance_' + op] = func(balances)
        else:
            features['closingBalance_' + op] = np.nan
    
    return {'bank_data.' + k:v for k,v in features.items()}


def bank_name(my_list):
    """
    Create features from a list of bank_names.
    Parameters
    ----------
    my_list : list
        List of bank_names
    Returns
    -------
    dict
        Features created
    """
    features = {}
    
    for bank_name in bank_names:
        features[bank_name] = my_list.count(bank_name)
    return {'bank_name.' + k:v for k,v in features.items()}


def credit_data(credit_data):
    """
    Create features from a list of credit_data.
    Parameters
    ----------
    credit_data : list
        List of credit_data with several keys, the most importants are:
        - 'brand' (str list)
        - 'level' (str list)
        - 'creditLimit' (numeric list)
        - 'availableCreditLimit' (numeric list)
    Returns
    -------
    dict
        Features created
    """

    features = {}
    credit_data = [item for item in credit_data if item]
    items = {}
    
    for col in [
        "brand",
        "level",
        "creditLimit",
        # "balanceDueDate",
        # "minimumPayment",
        # "isLimitFlexible",
        # "balanceCloseDate",
        "availableCreditLimit",
        # "balanceForeignCurrency",
    ]:
        if credit_data:
            items[col] = [item.get(col) for item in credit_data]
        else:
            items[col] = []
            
    for brand in cd_brands:
        features['brand_' + brand] = items['brand'].count(brand)
    
    # TODO: sum of levels
    for level in cd_levels:
        features['level_' + level] = items['level'].count(level)
        features['level_sum'] = sum([cd_levels.get(item, 0) for item in items['level']])
        
    for op, func in numeric_ops.items():
        for item in ['creditLimit', 'availableCreditLimit']:
            items[item] = [value for value in items[item] if value]
            if len(items[item]) > 0:
                features[item + '_' + op] = func(items[item])
            else:
                features[item + '_' + op] = np.nan
            
    return {'credit_data.' + k:v for k,v in features.items()}


def execution_status(my_list):
    features = {}
    
    for execution_status in execution_statuses:
        features[execution_status] = my_list.count(execution_status)
    return {'execution_status.' + k:v for k,v in features.items()}


def marketing_name(my_list):
    """
    Create features from a list of marketing_names.
    It uses mn_providers, mn_card_level and mn_card_type 
    to create the features.
    
    Parameters
    ----------
    my_list : list
        List of marketing_names
    Returns
    -------
    dict
        Features created
    """
    features = {}
    my_list = [item.upper() for item in my_list if item]
    
    for provider in mn_providers:
        features['provider_' + provider] = any([provider in item for item in my_list])
        
    for level in mn_levels:
        features['level_' + level] = any([level in item for item in my_list])
        
    for company in mn_card_companies:
        features['company_' + company] = any([company in item for item in my_list])
        
    return {'marketing_name.' + k:v for k,v in features.items()}


def mfa(x):
    return {'has_mfa_bool': any(x), 'n_mfa': sum(x)}


def owner(my_list):
    """
    Create features from a list of owners.
    Parameters
    ----------
    my_list : list
        List of owners
    Returns
    -------
    dict
        Features created
    """
    owner_list = list(set([item.lower() for item in my_list if item]))
    return {'owner_list': owner_list, 'owner_count': len(owner_list)}


def process_bank_type(my_list):
    return {'bank_type.PERSONAL_BANK': my_list.count("PERSONAL_BANK")}


def process_bank_products(my_list):
    features = {}
    full_list = sum(my_list, [])
    
    for bank_product in bank_products:
        features[bank_product] = full_list.count(bank_product)
    
    return {'bank_products.' + k:v for k,v in features.items()}


def process_bank_products(my_list):
    features = {}
    full_list = sum(my_list, [])
    
    for bank_product in bank_products:
        features[bank_product] = full_list.count(bank_product)
    
    return {'bank_products.' + k:v for k,v in features.items()}


def process_connector_products(df):
    for bank_product in bank_products:
        df['bank_product.' + bank_product] = df['connector.products'].apply(lambda x: x.count(bank_product))
    df = df.drop(columns=['connector.products'])
    return df
    
    
def process_identity(df):
    for col in [
        "identity.emails",
        "identity.phone_numbers",
    ]:
        df[col] = df[col].apply(lambda x: x if type(x)==list else [])
        df[col + "_len"] = df[col].apply(len)
        # df[col + "_list"] = df[col].apply(lambda x: [x.get("value") for x in x if x.get("value")])
        df[col + "_Personal_type"] = df[col].apply(lambda x: [x.get("type") for x in x if x.get("type")].count("Personal"))
        df[col + "_Work_type"] = df[col].apply(lambda x: [x.get("type") for x in x if x.get("type")].count("Work"))

    df = df.drop(columns=["identity.emails", "identity.phone_numbers"])
    return df
    
    
def create_accounts_features(data_, metadata):
    """
    Process a dictionary of reports
    Parameters
    ----------
    data : dict
        Dictionary of reports
    Returns
    -------
    dict
        Dictionary with the processed reports
    """
    
    account_df = deepcopy(data_)[accounts_cols]
    
    account_df[account_cat_cols] = process_rare_values(account_df[account_cat_cols], metadata)
    
    account_df = pd.get_dummies(account_df, columns=account_cat_cols)
    
    # Process connector products
    account_df = process_connector_products(account_df)
    
    # Process identity
    account_df = process_identity(account_df)
    
    # Process accounts
    account_fseries = account_df["accounts"].apply(lambda x : process_accounts(x))
    account_df = pd.concat([account_df, account_fseries] , axis=1)
    account_df.drop(columns=["accounts"], inplace=True)
    
    return account_df

class OBEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,  threshold=0.05, threshold_method='quantile', min_samples=2, transactions=True):
        """
        Encoder for Open Banking data

        Parameters
        ----------
        threshold : float, optional
            The threshold to use for the top categories, by default 0.2
        method : str, optional
            The method to use to compute the threshold, could be 'quantile' or 'weights', by default 'quantile'
        min_samples : int, optional
            The minimum number of samples to keep a category, by default 2
        """
        assert threshold >= 0 and threshold <= 1, 'Threshold must be between 0 and 1'
        assert min_samples > 0, 'Min samples must be greater than 0'
        assert threshold_method in ['quantile', 'weights'], 'threshold_threshold_method must be either quantile or weights'

        # Parameters
        self.metadata = None
        self.features_names = None
        self.init_col_set = None
        self.threshold = threshold
        self.min_samples = min_samples
        self.threshold_method = threshold_method
        self.transactions = transactions
        
    def create_init_dict(self):
        self.init_dict = {}
        
        if self.transactions:
            for metric in metrics:
                for key in self.metadata.keys():
                    if type(key) == tuple:
                        for value in self.metadata[key]:
                            key_concat = '_'.join([str(i) for i in key])
                            value_concat = '_'.join([str(i) for i in value])
                            self.init_dict.update({f'{key_concat}_{metric}_{value_concat}': None})
                        # self.init_dict.update({f'{key_concat}_{metric}_{value[0]}_{value[1]}_RARE': None})
                    else:
                        for col in numerical_cols:
                            for value in self.metadata[key]['top']: 
                                self.init_dict.update({f'{key}_{col}_{metric}_{value}': None})
                            self.init_dict.update({f'{key}_{col}_{metric}_RARE': None})
                            
            for period in ['daily', 'weekly', 'monthly']:
                self.init_dict.update({f'{period}_average': None})
                for value in self.metadata['category']['top']:
                    self.init_dict.update({f'category_{value}_{period}_average': None})
                self.init_dict.update({f'category_RARE_{period}_average': None})
            
        
        for key in account_cat_cols:
            for value in self.metadata[key]['top']:
                self.init_dict.update({f'{key}_{value}': None})
            self.init_dict.update({f'{key}_RARE': None})
            
    def generate_metadata(self, X, transactions=False):
        
        metadata_cols = all_cols if transactions else account_cat_cols
        
        # Iterate over all categorical variables
        for col in metadata_cols:
            
            # Calculate most frequent categories for each categorical variable
            
            # By number of transactions
            if 'amount' in X.columns:
                self.metadata[col] = X[~X['amount'].isna()][col].value_counts().to_dict()
            else:
                self.metadata[col] = X[col].value_counts().to_dict()
            
            # By number of connections item_ids (should we change item_id to client_user_id?)
            # self.metadata[col] = X.groupby(col)['id'].nunique().to_dict()
        
        if transactions:
            list_of_cat_date_combinations_cols = create_col_combinations(self.metadata.keys())
            
            for col in list_of_cat_date_combinations_cols:
                self.metadata[tuple(col)] = X[~X['amount'].isna()][col].value_counts().to_dict()


    def set_top_rare_metadata(self):
        # Map infrequent values to RARE category
        self.metadata = dict(self.metadata)
        
        for key in self.metadata.keys():
            
            d = pd.DataFrame(dict(self.metadata[key]), index=[0]).T

            # Compute quantile threshold according to method
            if self.threshold_method == 'quantile':
                cut = d.quantile(self.threshold)
            elif self.threshold_method == 'weights':
                cut = self.threshold*d.sum()

            # Drop categories with less than min_samples to preserve generalization
            d = d[d>self.min_samples]

            # Drop categories below threshold
            top_cat_list = d[(d >= cut)].dropna().index.to_list()
            rare_cat_list = [v for v in self.metadata[key] if v not in top_cat_list]
            
            # Update metadata with the top categories
            self.metadata[key]['top'] = top_cat_list
            self.metadata[key]['rare'] = rare_cat_list
            
                
    
    
    def fit(self, X, y=None):
        
        # Initialize metadata
        self.metadata = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Process DataFrame
        
        X_ = copy.deepcopy(X)
        loan_created_at = X_['created_at']
        index = X_.index
        X_ = pd.json_normalize(X_['ob_data']).set_index(index)
        X_['loan_created_at'] = loan_created_at
        
        # Transactions metadata
        if self.transactions:
            X_expanded = build_transactions_df(X_)
            X_expanded = process_transactions_data(X_expanded)
            self.generate_metadata(X_expanded, transactions=True)

        # Account metadata
        self.generate_metadata(X_)
        
        # Set top and rare categories
        self.set_top_rare_metadata()
        # self.features_names = X_.columns.tolist()
        
        # Initialize dictionary of features
        self.create_init_dict()
        
        return self
    
    def transform(self, X):
        
        if self.metadata is None:
            raise ValueError('Run fit before transform')
        
        X_ = copy.deepcopy(X)
        loan_created_at = X_['created_at']
        index = X_.index
        X_ = pd.json_normalize(X_['ob_data']).set_index(index)
        X_['loan_created_at'] = loan_created_at
        
        #######################################################
        ################ Feature Engineering ##################
        #######################################################
        
        ##### Compute Features #####
        
        # Initiate Features DataFrame
        features_df = pd.DataFrame([self.init_dict]*len(X_), index=X_.client_user_id)
        #print('Transactions Features Initialized')
           
        # Account Features
        accounts_features_df = create_accounts_features(X_, self.metadata)
        #print('Created Account Features')
        
        # Transactions Features
        if self.transactions:
            X_expanded = build_transactions_df(X_)
            X_expanded = process_transactions_data(X_expanded)
            transactions_features_df = create_transactions_features(X_expanded, self.metadata)
        #print('Created Transactions Features')
        
        # Remove OOV columns
        if self.transactions:
            transactions_features_df = transactions_features_df[[col for col in transactions_features_df.columns if col in features_df.columns]]
        # accounts_features_df = accounts_features_df[[col for col in accounts_features_df.columns if col in features_df.columns]]
        #print('Removed OOV columns')
        
        # Updated df with computed features
        if self.transactions:
            features_df.loc[:, transactions_features_df.columns.tolist()] = transactions_features_df
        features_df.loc[:, accounts_features_df.columns.tolist()] = accounts_features_df
        #print('Updated df with computed features')
        
        # Rearrange index to original order
        features_df = features_df.loc[X_.client_user_id]
        
        # Fill transactions features Nans with 0
        features_df = features_df.fillna(0)
        #print('Filled transactions features NaNs with 0')
                
        return features_df


    def get_feature_names(self):
        return self.features_names

    def get_features_types(self):
        pass

    def get_vocabulary(self):
        return self.metadata


    