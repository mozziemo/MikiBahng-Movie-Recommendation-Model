import os
import sys
import time

from flask import Flask, render_template, request

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import importlib

importlib.reload(sys)


app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/select-movies')
def movie_selections():
    return render_template('select_movies_50.html')	


@app.route('/movie-recommendations', methods=['POST'])
def movie_recommender():
    try:
        ### Load MovieLens data
        file_directory = os.path.join(os.getcwd(), 'data_folder')
        ratings_df = pd.read_csv(os.path.join(file_directory, 'ml-latest-small_ratings.csv'))
        movies_df = pd.read_csv(os.path.join(file_directory, 'ml-latest-small_movies.csv'))

        ### Set data format
        ratings_df['userId'] = ratings_df['userId'].astype(np.int32)
        ratings_df['movieId'] = ratings_df['movieId'].astype(np.int32)
        ratings_df['rating'] = ratings_df['rating'].astype(np.float32)
        ratings_df.drop('timestamp', axis=1, inplace=True)

        movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric).astype(np.int32)
        
        ### Get online request form data (favorite movies list) given by a new user
        selected = request.form.getlist('check')

        ### Add new user's data to ratings_df: 
        ## step 1: transform new user's data into a dataframe
        #df = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
        new_user_id = int(ratings_df[-1:]['userId']) + 1  #user_id = 672
        new_user_rating = 5.0
        
        new_user_movie_ids = selected  # e.g., [u'1',u'34',u'111',u'64249',u'69122']

        new_user_data = []
        for movie_id in new_user_movie_ids:
            new_user_data.append([new_user_id, int(movie_id), new_user_rating])

        # new user data in dataframe format
        new_user_df = pd.DataFrame(data = new_user_data, 
                                   columns = ["userId", "movieId", "rating"])

        ## step 2: add new_user_df to ratings_df
        ratings_df_all = pd.concat([ratings_df, new_user_df], axis=0)


        ### Prep data for Singular Value Decomposition
        ## step 1: pivot ratings_df_all
        R_df = ratings_df_all.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
        ## step 2: de-mean the data (to center or normalize by each user's mean)
        R = R_df.to_numpy().astype(np.float32) # R = R_df.as_matrix()
        user_ratings_mean = np.mean(R, axis = 1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)


        ### Singular Value Decomposition
        U, S, Vt = svds(R_demeaned, k = 50)
        ## convert S to the diagonal matrix form
        S = np.diag(S)

        ### Make predictions for all users from the decomposed matrices
        predicted_ratings_all = np.dot(np.dot(U, S), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df_all = pd.DataFrame(predicted_ratings_all, columns = R_df.columns)

        
        ### Generate movie recommendations for a selected user
        def movie_recommender(movies_df, ratings_df_all_users, pred_df_all_users, userId, num_recommend=10):            
            ## user_index = userId - 1 because userId starts from 1, while row index starts from 0 
            user_index = userId - 1 # new_user_index = len(R_df) - 1
            pred_df_user_sorted = pred_df_all_users.iloc[user_index].sort_values(ascending=False)

            ## Get the user's ratings data and merge with movies_df (movie info)
            user_ratings = ratings_df_all_users[ratings_df_all_users.userId == (userId)]
            user_movie_ratings = user_ratings.merge(movies_df, how = 'left', left_on = 'movieId', 
                        right_on = 'movieId').sort_values(['rating'], ascending=False)

            ## Recommend movies with the highest predicted ratings that the user hasn't seen yet
            recommendations = movies_df[~movies_df['movieId'].isin(user_movie_ratings['movieId'])]\
                    .merge(pd.DataFrame(pred_df_user_sorted).reset_index(), 
                                how = 'left', left_on = 'movieId', right_on = 'movieId')\
                    .rename(columns = {user_index: 'predictions'})\
                    .sort_values('predictions', ascending = False)\
                    .iloc[:num_recommend, :-1]
                                
            return user_movie_ratings, recommendations


        new_user_movie_ratings, new_user_recommendations = movie_recommender(movies_df, ratings_df_all, 
                                                                        preds_df_all, new_user_id, 10)


        ## Create a movie recommendations table that can be displayed on the website
        with pd.option_context('display.max_colwidth', -1):
            recommendations_df = new_user_recommendations[['title','genres']] \
                                    .rename(columns = {'title' : 'Movie Title','genres' : 'Genres'})
            # shift the starting index value from 0 to 1
            recommendations_df.index = np.arange(1, len(recommendations_df) + 1)
            # create HTML-friendly table            
            recommendations_table = recommendations_df.to_html(classes='bluetable')

        return render_template('movie_recommendations.html', selected=selected, 
                recommended_movies=recommendations_table, new_user_id=new_user_id)	


    except Exception as e:
        time.sleep(1)
        # print(repr(e))
        error = str(e)
        return render_template("movie_selections_error.html", error=error)



if __name__ == "__main__":
	app.run()


