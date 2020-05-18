__author__ = 'Saeid SOHILY-KHAH'
"""
Recommendation Systems: Content-Based Filtering
"""
import ast
import numpy as np
import pandas as pd
from scipy import spatial
from termcolor import colored


# Load data
def load_data():
    '''
    Load movies and user ratings data
    :return: dataframe:
    '''
    movies = pd.read_csv('./dataset/movies.csv')
    # Drop useless features
    movies = movies.drop(['Unnamed: 0', 'year'], axis=1)

    ratings = pd.read_csv('./dataset/ratings.csv')
    ratings = ratings.drop('Unnamed: 0', axis=1) # drop useless features
    return movies, ratings


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load data
    movies, ratings = load_data()

    # Definition of new_user with the movies has watched and rated
    _user = pd.DataFrame([
        {'title': 'GoldenEye', 'rating': 4},
        {'title': 'Making a Murderer', 'rating': 4.5},
        {'title': 'Frozen Silence', 'rating': 5},
        {'title': "Cat and Mouse", 'rating': 1},
        {'title': 'Sudden Death', 'rating': 3},
        {'title': 'Dead Presidents', 'rating': 4}
    ])

    # Initialization
    n_recommendations = 10 # number of output movies recommendations

    # Convert the list of genres using one hot encoding
    for ind in movies.index:
        for i in range(len(ast.literal_eval(movies['genres'][ind]))):
            genre = ast.literal_eval(movies['genres'][ind])[i]
            movies.at[ind, genre] = 1

    movies = movies.fillna(0)
    movies = movies.drop('genres', axis=1)

    # Add movieId to _user data info (instead of title)
    movieId_user = pd.merge(movies[movies['title'].isin(_user['title'].tolist())], _user).drop('title', axis=1)
    movieId_user = movieId_user[['movieId', 'rating']]
    movieId_user.sort_values(by=['movieId'])
    movieId_user = movieId_user.reset_index(drop=True) # reset index

    # Filtering out the movies from the input
    genres_user = movies[movies['movieId'].isin(movieId_user['movieId'].tolist())]
    genres_user.sort_values(by=['movieId'])
    genres_user = genres_user.reset_index(drop=True) # reset index
    genres_user = genres_user.drop(['movieId', 'title'], axis=1)

    # Compute the _user's genre weights (profile of user) -> [recommend movies that satisfy user's preferences]
    profile_user = genres_user.transpose().dot(movieId_user['rating'])

    # Create movies genres dataframe
    movies_genres = movies.set_index(movies['movieId'])
    movies_genres = movies_genres.drop(['movieId', 'title'], axis=1) # drop useless features

    # Multiply the genres by the weights and then take the weighted average
    recommendation = ((movies_genres * profile_user).sum(axis=1)) / (profile_user.sum())
    recommendation = pd.DataFrame(recommendation, columns=['score'])
    recommendation['movieId'] = recommendation.index

    # Sort the recommendation dataframe to get top k
    recommendation = recommendation.sort_values(by='score', ascending=False)[:n_recommendations]

    # Summarize result
    recommendation_movies = movies.loc[movies['movieId'].isin(recommendation['movieId'].tolist())]
    print('Movie Recommendations based on the Content-based Filtering are:')
    for i in range(len(recommendation_movies.title.values)):
        print('\t', colored(recommendation_movies.title.values[i], 'green'))# title of movie recommendation