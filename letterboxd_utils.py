import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import unicodedata
import numpy as np

def get_user_diary(username):
    """Creates a DataFrame of a user's Letterboxd diary given their username

    Args: 
        username (string): Letterboxd username

    Returns:
        df (DataFrame): DataFrame containing the information in a user's diary (movies, ratings, etc.)
    """

    # Finding the max page number
    url = f'https://letterboxd.com/{username}/films/diary/'
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    pagination = soup.find('div', class_='pagination')
    if pagination:
        max_page = int(pagination.find_all('li', class_='paginate-page')[-1].text) 
    else:
        max_page = 1

    all_data = [] 
    current_month = None 

    for i in range(1, max_page + 1):
        url = f'https://letterboxd.com/{username}/films/diary/page/{i}'
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')

        month_watched = soup.find_all(class_='td-calendar')
        day_watched = soup.find_all(class_='td-day diary-day center')
        films = soup.find_all('h3', class_='headline-3 prettify')
        released_dates = soup.find_all(class_='td-released center')
        ratings = soup.find_all('span', class_='rating')

        film_slug_list = []

        for j in range(len(films)): 
            if month_watched:
                current_month = month_watched[j].get_text(strip=True) if month_watched[j].get_text(strip=True) else current_month
            
            # getting film title for genre lookup
            film_title = films[j].get_text(strip=True)
            film_slug = '-'.join(re.sub(r'\s*:\s*', '-', film_title).lower().split())

            # Convert non-ASCII characters to ASCII equivalents
            film_slug = unicodedata.normalize('NFKD', film_slug).encode('ascii', 'ignore').decode('ascii')
            film_slug = film_slug.replace('.', '') 
            #replace all non-alphanumeric characters with empty string aside from hyphens
            film_slug = re.sub(r'[^a-zA-Z0-9-]', '', film_slug)
            #remove duplicate hyphens like '--'
            keep_film_slug = re.sub(r'-+', '-', film_slug)
            film_slug_list.append(keep_film_slug)

            film_slug = re.sub(r'-+', '-', film_slug)
            #remove hyphens at the beginning or end of the string
            film_slug = film_slug.strip('-')
            
            # fetching genres using the name of the film
            genre_url = f'https://letterboxd.com/film/{film_slug}/genres/'
            genre_html = requests.get(genre_url).text
            genre_soup = BeautifulSoup(genre_html, 'html.parser')
            genres = genre_soup.find('div', class_='text-sluglist capitalize')
            
            if genres:
                genre_list = [genre.text for genre in genres.find_all('a', class_='text-slug')]
                genres_str = ', '.join(genre_list) # joining the genres with commas
                
            else:
                genres_str = None
            
            data = {
                'Month': current_month,
                'Day': day_watched[j].get_text(strip=True) if j < len(day_watched) else None,
                'Film': film_title,
                'Released': released_dates[j].get_text(strip=True) if j < len(released_dates) else None,
                'Ratings': ratings[j].get_text(strip=True) if j < len(ratings) else None,
                'Genres': genres_str 
            }
            all_data.append(data)
            
    df = pd.DataFrame(all_data)
    return df, film_slug_list

def clean_diary_data(df):
   # Function implementation here
    """Cleans the DataFrame containing Letterboxd diary data.

    Args:
        df (DataFrame): DataFrame from get_user_diary function.

    Returns:
        df (DataFrame): Cleaned DataFrame.
    """

    # extracting month and year into separate columns
    df[['Month', 'Year']] = df['Month'].str.extract('([a-zA-Z]+)(\d{4})')
    month_mapping = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df['Month'] = df['Month'].map(month_mapping)
    df['Year'] = df['Year'].astype(int)

    # Convert star ratings to numerical values (handle NaN)
    def convert_rating(rating_str):
        if not rating_str:
            return 0  # Replace NaN with 0
        full_stars = rating_str.count('★')
        half_star = 0.5 if '½' in rating_str else 0
        return full_stars + half_star

    df['Ratings'] = df['Ratings'].apply(convert_rating)

    return df

def remove_duplicate_movies(df):
    # Function implementation here
    """Removes duplicate movie entries from the DataFrame.

    Args:
        df (DataFrame): DataFrame with movie entries.

    Returns:
        df (DataFrame): DataFrame with duplicate movie entries removed. 
    """
    # Remove duplicates based on 'Film', 'Year', 'Month', and 'Day'
    df = df.drop_duplicates(subset=['Film', 'Year', 'Month', 'Day'], keep='first') 
    return df

def linear_regression_numpy(X, y, alpha=0.01):
    """
    Performs linear regression using NumPy.

    Args:
        X (NumPy array): Feature matrix (n_samples x n_features).
        y (NumPy array): Target variable (n_samples).

    Returns:
        tuple: A tuple containing the model coefficients (weights) and the intercept.
    """

    # Add a bias term (intercept) to the feature matrix
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    identity = np.eye(X_b.shape[1])  # Create an identity matrix
    w = np.linalg.inv(X_b.T @ X_b + alpha * identity) @ X_b.T @ y  # Regularization
    intercept = w[0]
    weights = w[1:]
    return weights, intercept

def predict_linear_regression(X, weights, intercept):
    """Makes predictions using the learned weights and intercept."""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term for prediction as well
    y_pred = X_b @ np.concatenate(([intercept], weights))  # Use weights to predict
    return y_pred

def accuracy_within_half_star(y_true, y_pred):
    """Calculates the accuracy within ± 0.5 stars."""
    within_range = np.abs(y_true - y_pred) <= 0.5
    accuracy = np.mean(within_range)  # Calculate percentage of True values
    return accuracy