import pandas as pd


train_dataset_path = 'datasets/fraudTrain.csv'
test_dataset_path = 'datasets/fraudTest.csv'

def load_train_dataset(is_dataframe=True):
    """
        params:
            is_dataframe: if this is true, return pandas dataframe, otherwise numpy 2D array

        return: pandas dataframe or numpy 2D-array
    """
    return _load_dataset(train_dataset_path, is_dataframe)

def load_test_dataset(is_dataframe=True):
    """
        params:
            is_dataframe: if this is true, return pandas dataframe, otherwise numpy 2D array

        return: pandas dataframe or numpy 2D-array
    """
    return _load_dataset(test_dataset_path, is_dataframe)

def _load_dataset(path, is_dataframe):
    df = pd.read_csv(path, index_col=0)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    x, y = df.drop(['is_fraud'], axis=1), df['is_fraud']
    if is_dataframe:
        return x, y
    else:
        return x.to_numpy(), y.to_numpy()

def load_job_cat_dataset():
    df = pd.read_csv('datasets/job_cat.csv', usecols=['job', 'cat'], encoding='cp949')
    df['cat'] = df['cat'].str.replace('금융', 'finance')
    df['cat'] = df['cat'].str.replace('법률', 'law')
    df['cat'] = df['cat'].str.replace('건설/기계/제조', 'construction/manufacturing')
    df['cat'] = df['cat'].str.replace('군인/공무원', 'military/government')
    df['cat'] = df['cat'].str.replace('교육', 'education')
    df['cat'] = df['cat'].str.replace('영업/광고', 'sales/marketing')
    df['cat'] = df['cat'].str.replace('경영/사무', 'business management')
    df['cat'] = df['cat'].str.replace('예술/디자인', 'art')
    df['cat'] = df['cat'].str.replace('방송/연예', 'entertainment')
    df['cat'] = df['cat'].str.replace('운동', 'athlete')
    df['cat'] = df['cat'].str.replace('서비스', 'customer service')
    df['cat'] = df['cat'].str.replace('운송', 'transport')
    df['cat'] = df['cat'].str.replace('의료', 'medicine')
    df['cat'] = df['cat'].str.replace('농수산업', 'agriculture/fishery')
    df['cat'] = df['cat'].str.replace('연구/과학', 'research/science')
    df['cat'] = df['cat'].str.replace('기타', 'etc')
    df['cat'] = df['cat'].str.replace('IT', 'IT')
    df.rename(columns={'cat': 'job_cat'}, inplace=True)
    return df
