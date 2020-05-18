__author__ = 'Saeid SOHILY-KHAH'
"""
Recommendation Systems: Collaborative Filtering
"""
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
        {'title': 'Dead Presidents', 'rating': 4},
        {'title': 'Nick of Time', 'rating': 4}
    ])

    # Initialization
    k = 50 # top k users that are most similar to the input _user
    n_recommendations = 10 # number of output movies recommendations

    # Merge/Filter movies according to the _user info
    movies_user = pd.merge(movies[movies['title'].isin(_user['title'].tolist())], _user).drop('title', axis=1)

    # Filter ratings according to the movies in new_user dataframe
    ratings_user = ratings[ratings['movieId'].isin(movies_user['movieId'].tolist())]

    # Groupby userIds
    ratings_user = ratings_user.groupby(['userId'])

    # Sort filtered ratings by the common movies with _user (top k)
    ratings_user = sorted(ratings_user, key=lambda x: len(x[1]), reverse=True)[:k]

    # Compute the similarities of users to input _user
    similarities = {}
    for user_id, user_id_groupby in ratings_user:
        movies_user = movies_user.sort_values(by='movieId') # sort moives_user by movisIds
        user_id_groupby = user_id_groupby.sort_values(by='movieId') # sort user_id_groupby by movisIds

        # filter common movies between _user and used_id
        movies_user_filtered = movies_user[movies_user['movieId'].isin(user_id_groupby['movieId'].tolist())]

        _user_rating = np.array(movies_user_filtered['rating']) # rating array of _user
        users_rating = np.array(user_id_groupby['rating']) # rating array of user_id

        similarities[user_id] = 1 - spatial.distance.cosine(_user_rating, users_rating) # similarity estimation

    # Convert the similarities dictionary to pandas dataframe
    similarities_df = pd.DataFrame.from_dict(similarities,
                                             orient='index', # the dictionary's keys should be dataframe rows
                                             columns = ['similarity']) # columns' name
    similarities_df['userId'] = similarities_df.index
    similarities_df.index = range(len(similarities_df)) # reindex dataframe

    # Select top K similar users to _user based of his ratings (tok 50)
    similarities_df = similarities_df.sort_values(by='similarity', ascending=False)[:k]

    # Merge ratings in similarities dataframe to get the rated movies for each similar neighbour
    similarities_df = similarities_df.merge(ratings, left_on='userId', right_on='userId', how='inner')

    # Compute weighted rating (i.e. similarity times user's ratings)
    similarities_df['w_rating'] = similarities_df['similarity'] * similarities_df['rating']

    # Recommendation dataframe
    # Compute sum of similarity and weighted rating groupby movieId
    recommendation = similarities_df.groupby('movieId').sum()[['similarity', 'w_rating']]
    recommendation.columns = ['sum_similarity', 'sum_w_rating']

    # Compute the weighted average score
    recommendation['score'] = recommendation['sum_w_rating'] / recommendation['sum_similarity']
    recommendation['movieId'] = recommendation.index

    # Sort the recommendation dataframe to get top k
    recommendation = recommendation.sort_values(by='score', ascending=False)[:n_recommendations]

    # Summarize result
    recommendation_movies = movies.loc[movies['movieId'].isin(recommendation['movieId'].tolist())]
    print('Movie Recommendations based on the Collaborative Filtering are:')
    for i in range(len(recommendation_movies.title.values)):
        print('\t', colored(recommendation_movies.title.values[i], 'green'), # title of movie recommendation
              '------', colored(recommendation_movies.genres.values[i], 'red')) # genre of movie recommendation