import numpy as np

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

def mean(data_array):
    return sum(data_array)/len(data_array)

def standard_deviation(data_array):
    return (sum([(x - mean(data_array))**2 for x in data_array])/len(data_array))**0.5


songs_matrix = np.array([[1,0.1, 0.1, 0.1],
                          [0.1, 0.98, 0.9, 0.98],
                          [0.1, 0.97, 0.92, 0.95],
                          [0.1, 0.99, 0.95, 0.96]])

user_likes = [1, 0.1, 0.1, 0.1]

new_song = [0.9, 0.1, 0.1, 0.1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def z_score(x, mean, std):
    return (x - mean)/std

# The idea here is that as the absolute z-score increases, we are getting further away from the song. Data is transformed such that smaller values means closer to the song score
def transform_z(z):
    return 1/( 1 + (abs(z)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_correlations(matrix, scores):
    z_scores_matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    correlations = [np.corrcoef(z_scores_matrix[:, i], scores)[0, 1] for i in range(matrix.shape[1])]
    return correlations

def prediction(matrix, correlations):
    z_scores_new_song = (new_song - matrix.mean(axis=0)) / matrix.std(axis=0)
    print("z_scores_new_song", z_scores_new_song)
    prediction = sum(z * c for z, c in zip(z_scores_new_song, correlations))
    print("prediciton: ", prediction)
    return sigmoid(prediction)


def compute():
    correlations = compute_correlations(songs_matrix, user_likes)
    print("correlations: ", correlations)
    predict_user_score = prediction(songs_matrix, correlations)
    print("Predicted Score: ", predict_user_score)



songs_matrix = np.array([[0.5, 0.5, 0.5, 0.5],
                          [0.62, 0.58, 0.55, 0.7],
                          [0.99, 0.99, 0.9, 0.92],
                          [0.05, 0.1, 0.1, 0.1]])

user_likes = [1, 1, 0, 0]

new_song = [0.5, 0.5, 0.1, 0.1] # issue with singular matrices


compute()



