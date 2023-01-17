"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "Movies EDA", "About Sigma AI"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine by Sigma AI')
        st.write('### Sigma AI Recommendation Engine based on the MovieLens dataset')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        st.header("The Task")
        st.write("Sigma AI's task was to recommend the top 10 movies to a user based on the 3 movies the user selected. We used both the content-based and collaborative-based filtering approaches. Both approaches have their tradeoffs which were noted.")
        st.header("1. The Content-based filtering method")
        st.write("In this method, we recommended the 10 movies to the users based on the similarity between the movies. The similarities were computed from a similarity matrix. To get the similarity between movies, we used the cosine similarity equation and selected the movies with the highest similarity scores. This task was somewhat peculiar because he had to select similarity scores for each movie, we had 3 and then pool these together to then reselect the movies with the highest similarity scores from the pool.")
        st.subheader("Challenges of this method")
        st.write("We noted that we couldn’t be able to take user peculiarities into question while crafting the similarity scores. For example, users might want movies based on some demographics like religion, culture, or age among users, and we couldn’t use these due to the limitations of the dataset.")
        st.header("2. The Collaborative-based filtering method")
        st.write("In this method, we first analyzed what users rated the 3 selected movies highly. Given these selected users, we did a similarity score for the users related to each of our initial selected users based on their movie ratings. We then pooled similar users. After pooling, we analyzed what movies the pool of similar users have rated highly. We then created a popularity poll of similar users’ movies. From the popularity poll, we recommended the ten most popular movies.")
        st.subheader("Challenges of this method")
        st.write("1. We could only pool similar users based on their rating, but we couldn’t pool similar users based on location, culture, or other demographics.")
        st.write("2. We noticed that some movies that were in the rating dataset were missing in the movies dataset. Therefore, this limited the ability to get some movie titles.")
        st.write("3. We encountered a startup problem in some cases. This is a situation where we had some movies, but no user has rated those movies. In this case, we recommended the ten most popular rated movies from the rating dataset.")
        st.header("Benefits of using our recommendation system")
        st.markdown("**Revenue:** Movie sites like Netflix can earn more revenue by using our system. User churn will be prevented and users will be encouraged to keep coming back when our system recommends movies they like to them.")
        st.markdown("**Customer satisfaction:** Customers who are recommended movies they love will keep using the sites that use our recommendation system. Repeat customers spread the word to others and this increases the reputation of the site using our system.")
        st.markdown("**Personalization:** Most movie watchers watch a movie based on the recommendation of a friend. We are trying to model that scenario. When we recommend movies using our system to movie watchers, we believe that we are modeling what their friends would have done. That is what our collaborative-based filtering algorithm seeks to do. With this, movie viewers would love the sites using our recommendation system. They would see those sites as a personal friend.")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitchst.
    if page_selection == "Movies EDA":
        st.title("Exploratory Data Analysis of the MovieLens Dataset")
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        st.write("Exploratory data analysis is a visual technique for analyzing data. It is used to discover trends in the data and check for assumptions and patterns using statistical summary and graphical representation.")
        st.header("General Analysis of ratings dataset")
        st.write("When we analyzed the rating dataset, we noted the following unique features:")
        st.write("1. There are 162541 unique users.")
        st.write("2. There are 48213 unique rated movies.")
        st.write("3. The ratings start at 0.5 for worse movie, and the maximum is 5.0 for best movie.")
        st.header("Distribution of movies and ratings")
        st.image('resources/imgs/ratiings_nos.png',use_column_width=True)
        st.write("The movie data set contains 62423 unique movies. On analysis of the movie dataset with the rating dataset, we saw that 23% (14210) of the movies were not rated. 20% (12537) had single ratings, while 42% (26265) were rated between 2 and 50 times. 4% (2223) were rated 51 to 100 times, while 8% (5171) were rated 101 to 1000 times. And 3% (2117) were rated more than 1000 times. Therefore, a large percentage of the movies were unrated. We believe most of these unrated movies are old movies that were released before the rating system started.")
        st.header("Average Rating")
        st.image('resources/imgs/average_rating.png',use_column_width=True)
        st.write("We noticed that the average rating is 3.5 while the modal rating is 4.0. Therefore, we can conclude that movie viewers tend to be kind toward movie producers. They tend to rate movies highly. That means they generally like most of the films being produced. This is because film production quality keeps improving daily, and technology is also improving for movie production.")
        st.header("Top 10 Movie Raters")
        st.image('resources/imgs/top_users.png',use_column_width=True)
        st.write("We see that some users love rating movies more than other users. User ID 72315 rated movies the most, doing so 12952 times, followed by User ID 80974, who rated movies 3680 times.")
        st.header("Distribution of ratings by movie genre")
        st.image('resources/imgs/movie_genre.png',use_column_width=True)
        st.write("Users tend to rate movies in the drama genre more than all other genres. The comedy genre follows this. The IMAX genre is the least rated category of all the movies.")
        st.header("Movie production per year")
        st.image('resources/imgs/production_peryear.png',use_column_width=True)
        st.write("From the visual above, we see that more movies were produced in the 21st century (2000+) than in the 20th century (1900+). This is due to technological advancement and improved methods of production.")
        st.header("Movie ratings per year")
        st.image('resources/imgs/ratings_peryear.png',use_column_width=True)
        st.write("The image above shows that rating feedback was highest from 1994-2005. Lower rating feedback was provided for movies produced between 1937-1992. This could be related to the movie quality and visuals. Movies produced between 1900 and 1980 had low audio and graphical quality compared to movies produced in the 21st century. It seems most users do not rate movies because they do not like the image quality.")
        st.header("Post-1990 rating feedback.")
        st.image('resources/imgs/post1990_feedback.png',use_column_width=True)
        st.write("From what we noticed above, the higher rating feedback was recorded post-1995. We wanted to have a granular look at the rating feedback post-1990. From the image above, it is clear that no feedback was given for movies produced before 1995. We don’t know the reason why. However, there was one feedback for the year 1995. The year with the most rating feedback was 2016.")
        st.header("Frequently used tags")
        st.image('resources/imgs/tags.png',use_column_width=True)
        st.write("From the word cloud above, we noticed resonating and often-occurring tags in movie titles. They are love, story, murder, family, topless, and crime. We could also add drugs, relationships, comedy, and woman to that list. So I guess if a film producer wants his movie to be well rated, they should use one of these words in their movie titles.")
    if page_selection == "About Sigma AI":
        st.title("About Sigma AI")
        st.write("Sigma AI is Africa’s first company to use cutting-edge machine learning and artificial intelligence technology to produce recommendation systems for the entertainment, educational, book, and commercial industries. The recommendation systems take item-to-item and user idiosyncrasies into account.")
        st.write("Sigma AI is headquartered in Lagos, Nigeria, with offices in Johannesburg, South Africa, and Nairobi, Kenya. It comprises a core team of five machine learning engineers and software engineers. Some of our clients include Netflix, Facebook, Google, and DStv.")
        st.header("The Sigma AI Team")
        st.write("Here are the five core members of the Sigma AI team.")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image('resources/imgs/tony.jpg', caption="Tony Makwe.\n Project Team Lead")
        with col2:
            st.image('resources/imgs/david.jpg', caption="David Odimegwu.\n Technical Lead")
        with col3:
            st.image('resources/imgs/rabe.jpg', caption="Rabelani Ratshisuka.\n Administrative Lead")
        with col4:
            st.image('resources/imgs/basheer.jpg', caption="Basheer Ashafa.\n Project Manager")
        with col5:
            st.image('resources/imgs/nana.jpg', caption="Nana Adewale.\n Technical Support")
        
        




if __name__ == '__main__':
    main()
