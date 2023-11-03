import numpy as np
from math import sqrt

#Danceability
#Energy
#Tempo
#Year

song_example = [ 0.75, #Danceability
           0.68, #Energy
           0.60, #Tempo
           0.97]  #Year


songs_matrix = np.array([[0.37, 0.95, 0.73, 0.6 ],
 [0.16, 0.16, 0.06, 0.87],
 [0.6,  0.71, 0.02, 0.97],
 [0.83, 0.21, 0.18, 0.18],
 [0.3,  0.52, 0.43, 0.29],
 [0.61, 0.14, 0.29, 0.37],
 [0.46, 0.79, 0.2,  0.51],
 [0.59, 0.05, 0.61, 0.17],
 [0.07, 0.95, 0.97, 0.81],
 [0.3,  0.1,  0.68, 0.44]])

user_likes = [ 0.1, 0.5, 0.9, 0.2, 0.4, 0.6, 0.1, 0.1, 0.2, 0.9 ]
new_song = [0.3, 0.7, 0.2, 0.9]


def predict_score(songs_matrix, user_likes, new_song, bias):

    X_bias = [[bias] + song for song in songs_matrix]

    X = np.array(X_bias)
    y = np.array(user_likes)

    print("X: ", X)
    print("y: ", y)
    
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) #linalg (linear algebra) - using normal equation: https://www.datacamp.com/tutorial/tutorial-normal-equation-for-linear-regression

    coefficents = theta[1:]  # coefficents are the all but last elements
    intercept = theta[-1] # intercept is the last element 

    print("coefficents: ", coefficents)
    print("intercept: ", intercept)

    predicted_score = np.dot(theta, new_song)

    return predicted_score

#all 
songs_matrix2 = np.array([[1,0.1, 0.1, 0.1],
                          [0.1, 0.98, 0.9, 0.98],
                          [0.1, 0.97, 0.92, 0.95],
                          [0.1, 0.99, 0.95, 0.96]])


user_likes2 = [1, 0.1, 0.1, 0.1]

new_song2 = [0.9, 0.1, 0.1, 0.1] # issue with singular matrices

# print(predict_score(songs_matrix, user_likes, new_song, 0))
print(predict_score(songs_matrix2, user_likes2, new_song2, 0))

songs_matrix3 = (
    (0.5, 0.5, 0.5, 0.5),
    (0.59, 0.51, 0.55, 0.7),
    (0.99, 0.99, 0.9, 0.92),
    (0.05, 0.1, 0.1, 0.1),
    (0.72, 0.38, 0.91, 0.44),
    (0.58, 0.99, 0.37, 0.26)
)
user_likes2 = [10, 10, 1, 1, 6, 7]
new_song2 = [0.5, 0.5, 0.8, 0.1]

print('here')
scores = (a for a in zip(*songs_matrix3))
print(scores)


def count_up(scores, likes):
    table = []
    for i in enumerate(scores):
        table.append([])
        print(i[1])
        for j in enumerate(i[1]):
            for k in range(likes[j[0]]):
                table[i[0]].append(i[1][j[0]])
    return table


scores_table = count_up(scores,user_likes2)
print('scores table')
print(scores_table)


def get_means(table):
    return [sum(a)/len(a) for a in table]


means = get_means(scores_table)
print('means')
print(means)


def get_stdevs(table, means):
    return [sum([(score-mean[1])**2 for score in table[mean[0]]])/len(user_likes2) for mean in enumerate(means)]


stdevs = get_stdevs(scores_table, means)
print('stdevs')
print(stdevs)


def normal_distribution(mean , sd, x):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


expected_ratings = np.array([
    normal_distribution(means[i[0]], stdevs[i[0]], i[1])/(stdevs[i[0]]*sqrt(2*np.pi))*8 for i in enumerate(new_song2)
])     # this *8 normalises results to s zero to ten scale

print('expected ratings')
print(expected_ratings)
