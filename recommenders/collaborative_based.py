"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import heapq
from numpy.linalg import norm


# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# build utility matrix for users 
util_matrix = ratings_df.pivot_table(index=['userId'], columns=['movieId'], values='rating')
# Normalize each row (a given user's ratings) of the utility matrix
util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
# Fill Nan values with 0's, transpose matrix, and drop users with no ratings
util_matrix_norm.fillna(0, inplace=True)
# Remove rows with all 0s in a Dataframe
util_matrix_norm = util_matrix_norm.loc[(util_matrix_norm != 0).any(axis=1)]

# this function converts movies ids to movie titles
def indices_to_titles(idx_list):
    """
    function converts movie ids to movie titles. 
    parameters:
    idx_list: a list containing movie ids
    output:
    title_list: a list containing corresponding movie titles
    """
    title_list = []
    for idx in idx_list:
        movie_title = movies_df[movies_df['movieId'] == idx]['title']
        if len(movie_title) != 0:
            movie_title = movie_title.values[0]
            title_list.append(movie_title)        
    return title_list    

# for each of the movies, select the users with the highest ratings
def highest_rated_users(movie_list):
    """
    Given a movie title list, function returns users that are associated with that movie. For the sake of brevity
    only highest rating users are returned because they liked the movie best. 
    parameters:
    movie_list: list containing movie titles  
    output:
    users_raters_list: a list containing the user ids of users who rated a movie highest.   
    """

    # get the movie id of the rated data
    users_raters_list = []
    for movie in movie_list:
        movie_id = movies_df[movies_df['title'] == movie]['movieId'].values[0]
        rating = ratings_df[ratings_df['movieId'] == movie_id]['rating']
        # if the rating is not empty series, get the max rating
        if len(rating) != 0: 
            max_rating = rating.max()
            # get all users with this max_rating and given movie_id
            max_rating_users = ratings_df[(ratings_df['movieId'] == movie_id) & (ratings_df['rating'] == max_rating)]['userId'].tolist()
            users_raters_list = users_raters_list + max_rating_users
    return users_raters_list   


# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # get the list of users who highly rated the selected movies
    # startup problem:
    # if no users, we recommend the top-n most popular movies in the catalog
    users_list = highest_rated_users(movie_list)
    if len(users_list) == 0:
        top_movies =  ratings_df.groupby('movieId').mean().sort_values(by='rating', ascending=False).index[:top_n].to_list()
        top_n_titles = indices_to_titles(top_movies)
        return top_n_titles
    # from here, we have a populated users list
    # select an initial refetrence user
    collected_scores = []
    for user in users_list:
        temp_scores = []
        # extract similarity value between the reference user and all users excluding same
        # first get the users vector
        ref_vector = util_matrix_norm.iloc[user].to_numpy()
        # then get the vector for all the users and do similarity computation
        for index, rows in util_matrix_norm.iterrows():
            if index != user:
                subset_user_vector = rows.to_numpy()
                sim_score = np.dot(ref_vector, subset_user_vector)/(norm(ref_vector)*norm(subset_user_vector))
                temp_scores.append((sim_score, user))
        # sort the large list to the first 20
        twenty_highest = heapq.nlargest(20, temp_scores, key=lambda t: t[0])
        collected_scores = collected_scores + twenty_highest
    # we now sort the collected scores from all the users again to give the top_n
    collected_scores.sort(key=lambda x : x[0], reverse=True)
    # in collected scores, we have repeated users because we are dealing with several reference users
    # we now collect all the users
    selected_users = []
    for score, user in collected_scores: 
        if user not in selected_users:
            selected_users.append(user)
    # collect the top rated items of the selected users
    selected_movie_ids = []       
    for user in selected_users:
        user_df = ratings_df[ratings_df['userId'] == user].sort_values(by='rating', ascending=False)
        idx_movies = user_df['movieId'].values[:20].tolist()
        selected_movie_ids = selected_movie_ids + idx_movies
    # now we will do a tally of popularity of the movie_ids
    fav_movie_dict = {}
    for movie in selected_movie_ids:
        if movie in fav_movie_dict:
            fav_movie_dict[movie] += 1
        else:
            fav_movie_dict[movie] = 1
    # we now sort the favorite movies and select the ten best
    sorted_favs = sorted(fav_movie_dict.items(), key=lambda x: x[1], reverse=True)
    #top_movies = sorted_favs[:top_n]
    top_movies_list = []
    for movieid, freq in sorted_favs:
        top_movies_list.append(movieid)    
    top_n_titles2 = indices_to_titles(top_movies_list)
    return top_n_titles2[:top_n]    


    