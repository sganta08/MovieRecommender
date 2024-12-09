import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import unicodedata
import numpy as np
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import requests
import gzip
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def clean_diary_data(df):
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
    # Remove duplicates based on 'Film', 'Year', 'Month', and 'Day'
    df = df.drop_duplicates(subset=['Film', 'Year', 'Month', 'Day'], keep='first') 
    return df

def get_random_movies_for_prediction(api_key, num_movies=100, used_titles=None):
    """Get random movies from TMDB that have IMDb ratings in our dataset."""
    base_url = "https://api.themoviedb.org/3"
    movies = []
    page = 1
    max_pages = 50
    used_titles = set() if used_titles is None else set(used_titles)

    # Load IMDb ratings
    imdb_ratings = {}
    print("Loading IMDb ratings...")
    with gzip.open('title.ratings.tsv.gz', 'rt') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            imdb_ratings[row['tconst']] = float(row['averageRating'])
    
    print("Fetching movies...")
    while len(movies) < num_movies and page <= max_pages:
        params = {
            'api_key': api_key,
            'language': 'en-US',
            'sort_by': 'vote_count.desc',
            'page': page,
            'include_adult': 'false'
        }
        
        try:
            response = requests.get(f"{base_url}/discover/movie", params=params)
            data = response.json()
            
            for movie in data['results']:
                if len(movies) >= num_movies:
                    break
                    
                if movie['title'] in used_titles:
                    continue
                
                # Get detailed movie info
                movie_url = f"{base_url}/movie/{movie['id']}"
                movie_params = {
                    'api_key': api_key,
                    'append_to_response': 'external_ids'
                }
                movie_response = requests.get(movie_url, params=movie_params)
                movie_details = movie_response.json()
                
                # Get IMDb ID and rating
                imdb_id = movie_details.get('external_ids', {}).get('imdb_id')
                if imdb_id and imdb_id in imdb_ratings:
                    movies.append({
                        'Title': movie['title'],
                        'TMDB Rating': movie['vote_average'],
                        'TMDB Vote Count': movie['vote_count'],
                        'Popularity': movie['popularity'],
                        'Budget': movie_details.get('budget', 0),
                        'Revenue': movie_details.get('revenue', 0),
                        'Genres': ', '.join(g['name'] for g in movie_details.get('genres', [])),
                        'IMDb Rating': imdb_ratings[imdb_id]
                    })
                    used_titles.add(movie['title'])
                    print(f"Found {len(movies)}/{num_movies} movies", end='\r')
                
                time.sleep(0.25)
                
            page += 1
            
        except Exception as e:
            print(f"\nError on page {page}: {str(e)}")
            continue
    return pd.DataFrame(movies)

def get_movie_recommendations(api_key, trained_model, scaler, used_titles, num_movies=100):
    # Get random movies
    print("Fetching random movies...")
    random_movies = get_random_movies_for_prediction(api_key, num_movies, used_titles=used_titles)
    
    if random_movies.empty:
        print("No movies found!")
        return pd.DataFrame()
    
    # Process features using our existing function
    processed_data, _ = engineer_features_for_knn(random_movies)
    
    # predictions
    predictions = trained_model.predict(processed_data)
    
    random_movies['Predicted_Rating'] = predictions
    
    # Sort and filter recs
    recommendations = random_movies.sort_values('Predicted_Rating', ascending=False)
    loves = recommendations[recommendations['Predicted_Rating'] >= 4.0]
    
    return loves[['Title', 'Predicted_Rating', 'TMDB Rating', 'IMDb Rating']]

def engineer_features_for_knn(df):
    data = df.copy()
    
    # Keeping core features plus budget/revenue
    core_features = [
        'TMDB Rating',
        'IMDb Rating',
        'TMDB Vote Count',
        'Popularity',
        'Budget',
        'Revenue'
    ]
    
    # Filling missing budget/revenue with medians
    data['Budget'] = data['Budget'].fillna(data['Budget'].median())
    data['Revenue'] = data['Revenue'].fillna(data['Revenue'].median())
    
    data['Rating_Avg'] = (data['TMDB Rating'] + data['IMDb Rating']) / 2
    data['ROI'] = (data['Revenue'] - data['Budget']) / data['Budget'].where(data['Budget'] > 0, 1)
    
    # Create simplified genre features (main genres only)
    main_genres = ['Action', 'Drama', 'Comedy', 'Horror', 'Romance']
    for genre in main_genres:
        data[f'Genre_{genre}'] = data['Genres'].fillna('').str.contains(genre).astype(int)
    
    # Final feature list
    features_to_keep = (core_features + 
                       ['Rating_Avg', 'ROI'] + 
                       [f'Genre_{g}' for g in main_genres])
    
    data = data[features_to_keep]
    
    # Scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    data = pd.DataFrame(scaled_data, columns=data.columns)
    
    return data, scaler

def train_and_evaluate_knn(X, y, n_neighbors=15):
    """Train and evaluate KNN model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights='distance',
        metric='manhattan'
    )
    knn.fit(X_train, y_train)
    
    # predictions
    y_test_pred = knn.predict(X_test)
    
    # basic metrics
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    acc_within_half = np.mean(np.abs(y_test - y_test_pred) <= 0.5)
    acc_within_one = np.mean(np.abs(y_test - y_test_pred) <= 1.0)
    
    print(f"MSE: {mse:.3f}")
    print(f"R squared: {r2:.3f}")
    print(f"Accuracy (plus minus 0.5 stars): {acc_within_half:.3f}")
    print(f"Accuracy (plus minus 1 star): {acc_within_one:.3f}")
    
    return knn

def accuracy_within_half_star(y_true, y_pred):
    within_range = np.abs(y_true - y_pred) <= 0.5
    accuracy = np.mean(within_range)  # Calculate percentage of True values
    return accuracy

def search_movie(title, year, api_key):
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

def get_movie_details(movie_id, api_key):
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

def get_movie_data(details, title):
    """Extract only the most relevant features for prediction."""
    return {
        'Original Title': title,
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

def get_tmdb_data(movies_df, api_key, delay=0.25):
    all_data = []
    total_movies = len(movies_df)
    
    for idx, (_, row) in enumerate(movies_df.iterrows(), 1):
        title = row['Film']
        year = int(row['Released']) if 'Released' in movies_df.columns else None
        
        print(f"Processing {idx}/{total_movies}", end='\r')
        
        search_result = search_movie(title, year, api_key)
        if not search_result:
            print(f"\nCould not find movie: {title}")
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
    Note: As I mention in the notebook files, we need to download the IMDb ratings dataset first.
    """
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

    film_slug_list = []

    for i in range(1, max_page + 1):
        url = f'https://letterboxd.com/{username}/films/diary/page/{i}'
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')

        month_watched = soup.find_all(class_='td-calendar')
        day_watched = soup.find_all(class_='td-day diary-day center')
        films = soup.find_all('h3', class_='headline-3 prettify')
        released_dates = soup.find_all(class_='td-released center')
        ratings = soup.find_all('span', class_='rating')

        for j in range(len(films)): 
            if month_watched:
                current_month = month_watched[j].get_text(strip=True) if month_watched[j].get_text(strip=True) else current_month
            
            # getting film title for genre lookup
            film_title = films[j].get_text(strip=True)
            film_link = films[j].find('a')['href']  # Get the href attribute
            # Extract the slug from the href (format is "/username/film/film-slug/")
            film_slug = film_link.split('/')[3]  # Get the fourth element after splitting
                
            film_slug_list.append(film_slug)
            
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

def plot_model_analysis(model, X, y):
    """Create key visualization plots."""
    # Split data for val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Plot 1 is Predicted vs Actual ratings
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    y_pred = model.predict(X_val)
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Predicted vs Actual Ratings')
    
    # Plot 2 is for the error Error Distribution
    plt.subplot(1, 3, 2)
    errors = y_val - y_pred
    plt.hist(errors, bins=20, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    
    # Plot 3 is Accuracy by rating ranges
    plt.subplot(1, 3, 3)
    accuracy_by_range = []
    rating_ranges = [(1,2), (2,3), (3,4), (4,5)]
    
    for low, high in rating_ranges:
        mask = (y_val >= low) & (y_val < high)
        if mask.any():
            acc = np.mean(np.abs(y_val[mask] - y_pred[mask]) <= 0.5)
            accuracy_by_range.append(acc)
    
    plt.bar([f"{l}-{h}" for l,h in rating_ranges], accuracy_by_range)
    plt.xlabel('Rating Range')
    plt.ylabel('Accuracy (plus minus 0.5)')
    plt.title('Accuracy by Rating Range')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'overall_accuracy': np.mean(np.abs(y_val - y_pred) <= 0.5),
        'mse': mean_squared_error(y_val, y_pred),
        'accuracy_by_range': dict(zip([f"{l}-{h}" for l,h in rating_ranges], accuracy_by_range))
    }