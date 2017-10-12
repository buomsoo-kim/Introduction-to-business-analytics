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

# pandas의 as_matrix() 함수로 dataframe을 numpy array로 바꾼다
ratings_train = ratings_train.as_matrix(columns = ['user id', 'movie id', 'rating'])
ratings_test = ratings_test.as_matrix(columns = ['user id', 'movie id', 'rating'])

# numpy array로 바꾸어도 shape는 그대로 유지된다
# print(ratings_train.shape)
# print(ratings_test.shape)

users_list = []
for i in range(1, users.shape[0]):
    temp = []
    for j in range(0, len(ratings_train)):
        if ratings_train[j][0] == i:
            temp.append(ratings_train[j])
        else:
            break
    ratings_train = ratings_train[j:]
    users_list.append(temp)

# print(len(users_list))
# print(users_list)

def EucledianScore(train_user, test_user):
    s = 0
    count = 0
    for i in test_user:
        score = 0
        for j in train_user:
            if(int(i[1]) == int(j[1])):
                score= ((float(i[2])-float(j[2]))*(float(i[2])-float(j[2])))
                count= count + 1
            s = s + score
    if(count<4):
        s = 1000000
    return(math.sqrt(s))

score_list = []
for i in range(942):
    score_list.append([i+1, EucledianScore(users_list[i], ratings_test)])

score = pd.DataFrame(score_list, columns = ['user id', 'Euclidean Score'])
score = score.sort_values(by = 'Euclidean Score')
score.reset_index()

# print(score.shape)
# print(score.head())    # 유사도가 가장 높은 5명을 출력
# print()
# print(score.tail())    # 유사도가 가장 낮은 5명을 출력

# pandas dataframe을 numpy array로 변환한다
score_matrix = score.as_matrix()

# user 310이 평가한 모든 영화 정보를 담기 위한 full_list와
# user 943과 user 310이 공히 평가한 영화 정보를 담기 위한 common_list를 생성한다
user = int(score_matrix[0][0])
common_list = []
full_list = []

# common_list와 full_list를 채워넣는다
for i in ratings_test:
    for j in users_list[user-1]:
        if int(i[1]) == int(j[1]):
            common_list.append(int(j[1]))
        full_list.append(j[1])

# 각 리스트를 집합으로 변환한다
common_list = set(common_list)
full_list = set(full_list)

# 추천 영화를 추려내기 위해 user 310이 평가한 영화 중에
# user 943이 이미 본 영화들을 배제한다
recommendation = full_list.difference(common_list)    # recommendation = full_list - common_list

item_list = (((pd.merge(items, ratings).sort_values(by = 'movie id')).groupby('movie title')))['movie id', 'movie title', 'rating']
item_list = item_list.mean()

# print(item_list)

item_list['movie title'] = item_list.index
item_list = item_list.as_matrix()

recommendation_list = []
for i in recommendation:
    recommendation_list.append(item_list[i-1])
    
# 추천 영화를 평균 평점(mean rating)이 높은 순으로 출력한다
recommendation = (pd.DataFrame(recommendation_list, columns = ['movie id', 'mean rating', 'movie title'])).sort_values(by = 'mean rating', ascending = False)
print(recommendation[['mean rating', 'movie title']])    # 추천 영화의 평균 평점과 제목을 출력한다
