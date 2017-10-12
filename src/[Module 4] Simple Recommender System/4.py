import pandas as pd
import numpy as np

# 각 데이터(user, item, rating)의 열 이름(column name)을 정의한다
user_cols = ['user id','age','gender','occupation','zip code']
item_cols = ['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western']
rating_cols = ['user id','movie id','rating','timestamp']

# pandas의 read_csv() 함수를 활용해 각 데이터셋을 불러온다
users = pd.read_csv('ml-100k/u.user', sep = '|', names = user_cols, encoding = 'latin-1')
items = pd.read_csv('ml-100k/u.item', sep = '|', names = item_cols, encoding = 'latin-1')
ratings = pd.read_csv('ml-100k/u.data', sep = '\t', names = rating_cols, encoding = 'latin-1')

whole_data = pd.merge(pd.merge(items, ratings), users)    # items, ratings, users 데이터 셋을 합친다
print(whole_data.head())    # whole_data의 첫 5 행을 출력한다

ratings_total = whole_data.groupby('movie title').size()    # number of people who rated each movie
print(ratings_total.head())    # ratins_total의 첫 5행을 출력한다

ratings_mean = (whole_data.groupby('movie title'))['movie title', 'rating'].mean()    # mean rating of each movie
print(ratings_mean.head())    # ratings_mean의 첫 5행을 출력한다

ratings_total = pd.DataFrame({'movie title': ratings_total.index, 'total ratings': ratings_total.values})   # ratings_total을 데이터 프레임으로 변환한다

ratings_mean['movie title'] = ratings_mean.index   # ratings_mean의 'movie title'열을 정의한다

final = pd.merge(ratings_mean, ratings_total).sort_values(by = 'total ratings', ascending= False)    # ratings_mean과 ratings_total 데이터 프레임을 합친다
print(final.head())    # final의 첫 5행을 출력한다

final = final[:300].sort_values(by = 'rating', ascending = False)    # 많은 rating을 가진 영화 순서대로 300개를 선택하고 평균 평점이 높은 순서로 정렬한다
print(final.head(10))    # 상위 10개 영화를 추천한다
