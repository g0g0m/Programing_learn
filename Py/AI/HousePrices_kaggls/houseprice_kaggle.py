# numpy , pandas
import numpy as np
from numpy.random.mtrand import standard_exponential 
import pandas as pd
from scipy.sparse.construct import random
# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# 可視化用ライブラリ
import matplotlib.pyplot as plt
import seaborn as sns

#pandasのカラムが100列まで見れるようにする
pd.set_option('display.max_columns', 100)

#データの読み込み
train_df = pd.read_csv('/home/g0n/ドキュメント/Programing_git/Py/AI/HousePrices_kaggls/train1.csv',index_col=0)
test_df = pd.read_csv("/home/g0n/ドキュメント/Programing_git/Py/AI/HousePrices_kaggls/test1.csv")

# 売却価格の概要をみてみる
print(train_df["SalePrice"].describe())

all_df = pd.concat([train_df.drop(columns='SalePrice'), test_df])

num2str_list = ['MSSubClass','YrSold','MoSold']
for column in num2str_list:
    all_df[column] = all_df[column].astype(str)

# 変数の型ごとに欠損値の扱いが異なるため、変数ごとに処理
for column in all_df.columns:
    # dtypeがobjectの場合、文字列の変数
    if all_df[column].dtype=='O':
        all_df[column] = all_df[column].fillna('None')
    # dtypeがint , floatの場合、数字の変数
    else:
        all_df[column] = all_df[column].fillna(0)

# 特徴量エンジニアリングによりカラムを追加する関数
def add_new_columns(df):
    # 建物内の総面積 = 1階の面積 + 2階の面積 + 地下の面積
    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]

    # 一部屋あたりの平均面積 = 建物の総面積 / 部屋数
    df['AreaPerRoom'] = df['TotalSF']/df['TotRmsAbvGrd']

    # 築年数 + 最新リフォーム年 : この値が大きいほど値段が高くなりそう
    df['YearBuiltPlusRemod'] = df['YearBuilt']+df['YearRemodAdd']

    # お風呂の総面積
    # Full bath : 浴槽、シャワー、洗面台、便器全てが備わったバスルーム
    # Half bath : 洗面台、便器が備わった部屋)(シャワールームがある場合もある)
    # シャワーがない場合を想定してHalf Bathには0.5の係数をつける
    df['TotalBathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

    # 合計の屋根付きの玄関の総面積 
    # Porch : 屋根付きの玄関 日本風にいうと縁側
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

    # プールの有無
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    # 2階の有無
    df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    # ガレージの有無
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    # 地下室の有無
    df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
 
    # 暖炉の有無
    df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# カラムを追加
add_new_columns(all_df)

# pd.get_dummiesを使うとカテゴリ変数化できる。
all_df = pd.get_dummies(all_df)
all_df.head()

# 学習データと予測データに分割して元のデータフレームに戻す。
train_df = pd.merge(all_df.iloc[train_df.index[0]:train_df.index[-1]],train_df['SalePrice'],left_index=True,right_index=True)
test_df = all_df.iloc[train_df.index[-1]:]

train_df = train_df[(train_df['LotArea']<20000) & (train_df['SalePrice']<400000)& (train_df['YearBuilt']>1920)]

# SalePriceLogに対数変換した値を入れる。説明の都合上新たなカラムを作るが、基本的にそのまま代入して良い。
# np.log()は底がeの対数変換を行う。
train_df['SalePriceLog'] = np.log(train_df['SalePrice'])

train_x = train_df.drop(columns = ['SalePrice', 'SalePriceLog'])
train_y = train_df['SalePriceLog']

test_x = test_df

def lasso_turning(train_x, train_y):
    param_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    for cnt, alpha in enumerate(param_list):

        lasso = Lasso(alpha=alpha)

        pipeline = make_pipeline(StandardScaler(), lasso)

        x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

        pipeline.fit(x_train, y_train)

        train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(x_train)))
        test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(x_test)))

        if cnt == 0:
            best_score = test_rmse
            best_param = alpha
        elif best_score > test_rmse:
            best_score = test_rmse
            best_param = alpha

    print('alpha : ' + str(best_param))
    print('test score is : ' +str(round(best_score,4)))

    return best_param

best_alpha = lasso_turning(train_x, train_y)

lasso = Lasso(alpha = best_alpha)

pipeline = make_pipeline(StandardScaler(), lasso)

pipeline.fit(train_x, train_y)

pred = pipeline.predict(test_x)

# 400,000より高い物件は除去
pred_ex_outliars = pred[pred<400000]

# 学習データの住宅価格をプロット(外れ値除去済み)
sns.histplot(train_df['SalePrice'])
# 歪度と尖度
print(f"歪度: {round(pd.Series(train_df['SalePrice']).skew(),4)}" )
print(f"尖度: {round(pd.Series(train_df['SalePrice']).kurt(),4)}" )

submission_df = pd.read_csv('/home/g0n/ドキュメント/Programing_git/Py/AI/HousePrices_kaggls/sample_submission.csv')

submission_df['SalePrice'] = pred

# submission.csvを出力
submission_df.to_csv('submission.csv',index=False)

