import numpy as np
import pandas as pd
import os
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder

URL = 'https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1'
DIR = '../Data'
FILE = 'nba2k-full.csv'
TARGET = 'salary'
HIGH_CARDINALITY = 50
HIGH_CORR = 0.5


def get_data() -> str:
    fullpath = os.path.join(DIR, FILE)
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    if FILE not in os.listdir(DIR):
        print('Train dataset loading.')
        r = requests.get(URL, allow_redirects=True)
        open(fullpath, 'wb').write(r.content)
        print('Loaded.')
    return fullpath


def clean_data(path: str) -> pd.DataFrame:
    assert os.path.isfile(path), f"{path} is not a file"
    df = pd.read_csv(path)
    df.b_day = pd.to_datetime(df.b_day, format='%m/%d/%y')
    df.draft_year = pd.to_datetime(df.draft_year, format='%Y')
    df.team = df.team.fillna('No Team')
    df.height = df.height.apply(lambda s: s.split()[-1]).astype(float)
    df.weight = df.weight.apply(lambda s: s.split()[-2]).astype(float)
    df.salary = df.salary.apply(lambda s: s.removeprefix('$')).astype(float)
    df.country = df.country.mask(df.country != 'USA', other='Not-USA')
    df.draft_round = df.draft_round.mask(df.draft_round == 'Undrafted', other='0')
    return df


def feature_data(df: pd.DataFrame) -> pd.DataFrame:
    df.version = pd.to_datetime(df.version, format='NBA2k%y')
    df['age'] = df.version.dt.year - df.b_day.dt.year
    df['experience'] = df.version.dt.year - df.draft_year.dt.year
    df['bmi'] = df.weight / df.height ** 2
    df = df.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'])
    high_card_cols = [col for col in df.columns if df[col].dtype == object
                      and df[col].nunique() >= HIGH_CARDINALITY]
    df = df.drop(columns=high_card_cols)
    return df


def multicol_data(df: pd.DataFrame) -> pd.DataFrame:
    corr_matr = df.drop(columns=TARGET).corr(numeric_only=True).abs()
    collinear_features = (corr_matr[(corr_matr < 1) & (corr_matr > HIGH_CORR)]
                          .dropna(how='all').index)
    feature_drop = df[collinear_features].corrwith(df[TARGET]).idxmin()
    return df.drop(columns=feature_drop)


def transform_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    scaler = StandardScaler()
    num_features = scaler.fit_transform(df.select_dtypes('number').drop(columns=TARGET))
    df_num = pd.DataFrame(num_features, columns=scaler.feature_names_in_)
    encoder = OneHotEncoder(sparse_output=False)
    cat_features = encoder.fit_transform(df.select_dtypes('object'))
    df_cat = pd.DataFrame(cat_features, columns=np.concatenate(encoder.categories_))
    return df_num.join(df_cat), df[TARGET]


def main():
    data_path: str = get_data()
    data: pd.DataFrame = clean_data(data_path)
    data = feature_data(data)
    data = multicol_data(data)
    features, target = transform_data(data)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print(features.shape, target.shape)


if __name__ == '__main__':
    main()
