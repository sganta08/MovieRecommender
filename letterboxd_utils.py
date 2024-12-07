import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import unicodedata
import numpy as np
import time
from typing import List, Dict, Optional
from tqdm import tqdm
from datetime import datetime

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

def search_movie(title: str, year: Optional[int], api_key: str) -> Optional[Dict]:
    """
    Search for a movie by title and optionally year.
    
    Args:
        title: Movie title to search for
        year: Optional release year to help match the correct movie
        api_key: TMDB API key
        
    Returns:
        Dict containing movie search result or None if not found
    """
    base_url = "https://api.themoviedb.org/3"
    url = f"{base_url}/search/movie"
    params = {
        'api_key': api_key,
        'query': title,
        'language': 'en-US',
        'include_adult': 'true'  # Include all movies to ensure we find the right one even if adult
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get('results', [])
        
        if not results:
            return None
            
        if year:
            # Try to find exact year match first
            for result in results:
                release_date = result.get('release_date', '')
                if release_date and int(release_date[:4]) == year:
                    return result
                    
        return results[0]
        
    except Exception as e:
        print(f"Error searching for {title}: {str(e)}")
        return None

def get_movie_details(movie_id: int, api_key: str) -> Optional[Dict]:
    """
    Get detailed information about a movie.
    
    Args:
        movie_id: TMDB movie ID
        api_key: TMDB API key
        
    Returns:
        Dict containing detailed movie information or None if not found
    """
    base_url = "https://api.themoviedb.org/3"
    url = f"{base_url}/movie/{movie_id}"
    params = {
        'api_key': api_key,
        'language': 'en-US',
        'append_to_response': 'credits,keywords,external_ids'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting details for movie {movie_id}: {str(e)}")
        return None

def get_movie_data(details: Dict, original_title: str) -> Dict:
    """Extract only the most relevant features for prediction."""
    return {
        'Original Title': original_title,
        'TMDB ID': details.get('id'),
        'IMDb ID': details.get('external_ids', {}).get('imdb_id'),
        'Budget': details.get('budget', 0),
        'Revenue': details.get('revenue', 0),
        'TMDB Rating': details.get('vote_average'),
        'TMDB Vote Count': details.get('vote_count'),
        'Popularity': details.get('popularity'),
        'Genres': ', '.join(g['name'] for g in details.get('genres', [])),
        'Adult': details.get('adult', False)
    }

def get_tmdb_data(movies_df: pd.DataFrame, api_key: str, delay: float = 0.25) -> pd.DataFrame:
    """Get streamlined TMDB data for movies."""
    all_data = []
    
    for _, row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Fetching TMDB data"):
        title = row['Film']
        year = int(row['Released']) if 'Released' in movies_df.columns else None
        
        search_result = search_movie(title, year, api_key)
        if not search_result:
            print(f"Could not find movie: {title}")
            continue
            
        time.sleep(delay)
        
        details = get_movie_details(search_result['id'], api_key)
        if not details:
            continue
            
        time.sleep(delay)
        movie_data = get_movie_data(details, title)
        all_data.append(movie_data)
    
    return pd.DataFrame(all_data)

def fetch_imdb_ratings(tmdb_data):
    """
    Fetches IMDb ratings using the IMDb dataset.
    Note: You'll need to download the IMDb ratings dataset first.
    """
    import gzip
    import csv
    
    imdb_ratings = {}
    with gzip.open('title.ratings.tsv.gz', 'rt') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            imdb_ratings[row['tconst']] = {
                'rating': float(row['averageRating']),
                'votes': int(row['numVotes'])
            }
    
    ratings = []
    for imdb_id in tmdb_data['IMDb ID']:
        if imdb_id in imdb_ratings:
            ratings.append(imdb_ratings[imdb_id]['rating'])
        else:
            ratings.append(None)
            
    return ratings