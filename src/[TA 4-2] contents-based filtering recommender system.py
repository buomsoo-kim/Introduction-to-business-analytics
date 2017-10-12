import pandas as pd
import numpy as np
import math

# 각 데이터(user, item, rating)의 열 이름(column name)을 정의한다
user_cols = ['user id','age','gender','occupation','zip code']
item_cols = ['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western']
rating_cols = ['user id','movie id','rating','timestamp']

# pandas의 read_csv() 함수를 활용해 각 데이터셋을 불러온다
users = pd.read_csv('ml-100k/u.user', sep = '|', names = user_cols, encoding = 'latin-1')       
items = pd.read_csv('ml-100k/u.item', sep = '|', names = item_cols, encoding = 'latin-1')
ratings = pd.read_csv('ml-100k/u.data', sep = '\t', names = rating_cols, encoding = 'latin-1')

# print(users.shape)     # 943 users in total
# print(items.shape)     # 1682 items (i.e., movies) in total
# print(ratings.shape)   # 100000 ratings in total

# print(users.head())    # users dataframe의 첫 5 행의 데이터
# print(items.head())    # items dataframe의 첫 5 행의 데이타
# print(ratings.head())    # ratings dataframe의 첫 5 행의 데이타

ratings_train = (ratings.sort_values('user id'))[:99832]    
ratings_test = (ratings.sort_values('user id'))[99833:]

# print(ratings_train.shape)
# print(ratings_test.shape)

# print(ratings_train.head())
# print()
# print(ratings_train.tail())
# print()
# print(ratings_test.head())
# print()
# print(ratings_test.tail())

ratings_test = ratings_test.sort_values(by = ['rating'], ascending = False)    # rating을 기준으로 내림차순 정렬을 한다

# pandas의 as_matrix() 함수로 dataframe을 numpy array로 바꾼다
ratings_train = ratings_train.as_matrix(columns = ['user id', 'movie id', 'rating'])
ratings_test = ratings_test.as_matrix(columns = ['user id', 'movie id', 'rating'])

# numpy array로 바꾸어도 shape는 그대로 유지된다
# print(ratings_train.shape)
# print(ratings_test.shape)

# print(ratings_test)

num_recommendations = 5    # 추천하고 싶은 영화의 개수 - 사용자가 임의로 지정할 수 있다

favored_items = []
for i in range(num_recommendations):
    favored_items.append(ratings_test[i][1])
	
def EucledianDist(item1, item2):    # 두 영화 간의 거리를 계산하기 위한 함수를 정의한다
    s = 0
    v1 = item1[5:]
    v2 = item2[5:]
    
    for i in range(len(v1)):
        temp = (v1[i]-v2[i])*(v1[i]-v2[i])    # 각 원소를 서로 뺀 후에 제곱근 해 모두 더한다
        s += temp
    return math.sqrt(s)

items = items.as_matrix()    # items 데이터프레임을 numPy 배열로 변환한다

distance_matrix = np.zeros((1682, 1682))    # 1682X1682 크기의 0으로 이루어진 행렬을 생성한다

for i in range(len(items)):
    for j in range(len(items)):
        distance_matrix[i][j] = EucledianDist(items[i], items[j])    # 영행렬의 각 원소를 각 열번호와 행번호의 영화 간의 거리로 계산해 담는다

distance_matrix    # 각 영화 간의 거리를 나타내는 2차원 배열(행렬)

recommended_movies = []
for i in favored_items:
    idx = np.argmin(distance_matrix[i])          # 각 영화와 가장 거리가 짧은 영화의 인덱스를 찾는다
    recommended_movies.append(items[idx][1])     # 인덱스를 통해 그 영화의 이름을 찾아 리스트에 첨부한다

# 추천 영화를 출력한다
for movie in recommended_movies:
    print(movie)