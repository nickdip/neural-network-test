import numpy as np 

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

