"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import heapq
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer 

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
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
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000)
    # tfidf was added !!!!!
    # Instantiating and generating the count matrix
    tf = TfidfVectorizer(analyzer='word', min_df=0, stop_words='english', max_features = 10000)
    # Produce a feature matrix, where each row corresponds to a book,
    # with TF-IDF features as columns 
    count_matrix = tf.fit_transform(data['keyWords'])
    sparse_df = pd.DataFrame(count_matrix.toarray(), columns=tf.get_feature_names_out())
    # get the indices of the chosen movies
    movie_one_idx = indices[movie_list[0]]
    movie_two_idx = indices[movie_list[1]]
    movie_three_idx = indices[movie_list[2]]
    # do the similarity matrix for each movie and other movies
    final_scores = []
    for item in movie_list:
        temp_scores = []
        movie_idx = indices[item]
        # get the movie vector
        movie_vector = sparse_df.iloc[movie_idx].to_numpy()
        # get all the rows of the data frame
        for index, rows in sparse_df.iterrows():
            if movie_idx != index:
                # get the vector for that row
                ref_vector = rows.to_numpy()
                # do the similarity score
                sim_score = np.dot(ref_vector, movie_vector)/(norm(ref_vector)*norm(movie_vector)) 
                temp_scores.append((sim_score, index))
        # get the 10 highest ratings 
        five_highest = heapq.nlargest(20, temp_scores, key=lambda t: t[0])
        final_scores = final_scores + five_highest
    # sort the final scores to get the top 15 movies
    final_scores.sort(key=lambda x : x[0], reverse=True)
    # get the movie titles from the final_scores list and put in a list
    top_titles = [  indices[indices == indice].index[0] for scores, indice in final_scores]
    return top_titles[:top_n]
    