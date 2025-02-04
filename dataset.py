import os
import zipfile
import requests
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Constants
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "movielens_1m"
EXTRACTED_DIR = os.path.join(DATA_DIR, "ml-1m")
OUTPUT_DIR = "Amir"  # Change this if needed
SEQ_LENGTH = 10  # Max sequence length before splitting target


def download_and_extract():
    """Download and extract the MovieLens dataset if not already available."""
    zip_path = os.path.join(DATA_DIR, "ml-1m.zip")
    if not os.path.exists(zip_path):
        print("Downloading MovieLens dataset...")
        response = requests.get(MOVIELENS_URL, stream=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Download complete!")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete!")


def load_data():
    """Load ratings and movie metadata."""
    ratings_path = os.path.join(EXTRACTED_DIR, "ratings.dat")
    movies_path = os.path.join(EXTRACTED_DIR, "movies.dat")

    ratings = pd.read_csv(ratings_path, sep="::", engine="python",
                          names=["userId", "movieId", "rating", "timestamp"], encoding="ISO-8859-1")

    movies = pd.read_csv(movies_path, sep="::", engine="python",
                         names=["movieId", "title", "genres"], encoding="ISO-8859-1")

    return pd.merge(ratings, movies, on="movieId")


def filter_users(data, min_interactions=5, max_interactions=100):
    """Filter users with too few or too many interactions."""
    user_counts = data['userId'].value_counts()
    valid_users = user_counts[(user_counts >= min_interactions) & (user_counts <= max_interactions)].index
    return data[data['userId'].isin(valid_users)]


def filter_popular_movies(data, threshold=0.05):
    """Remove the most popular movies to improve generalization."""
    movie_counts = data['movieId'].value_counts()
    top_movies = movie_counts.nlargest(int(len(movie_counts) * threshold)).index
    return data[~data['movieId'].isin(top_movies)]


def time_based_split(data, val_count=2, test_count=2):
    """Use strict time-based splitting: keep the last 2 interactions for val and test."""
    train_data, val_data, test_data = [], [], []
    train_genres, val_genres, test_genres = [], [], []
    targets_train, targets_val, targets_test = [], [], []

    users = data['userId'].unique()
    for user in tqdm(users, desc='Splitting Users by Time'):
        user_data = data[data['userId'] == user].sort_values('timestamp')

        if len(user_data) < val_count + test_count + SEQ_LENGTH:
            continue  # Skip users with too few interactions

        train_subset = user_data.iloc[:- (val_count + test_count)]
        val_subset = user_data.iloc[-(val_count + test_count):-test_count]
        test_subset = user_data.iloc[-test_count:]

        # Extract sequences
        train_movies = train_subset['movieId'].tolist()
        val_movies = val_subset['movieId'].tolist()
        test_movies = test_subset['movieId'].tolist()

        train_genre = train_subset['genres'].tolist()
        val_genre = val_subset['genres'].tolist()
        test_genre = test_subset['genres'].tolist()

        if len(train_movies) > SEQ_LENGTH:
            train_data.append(train_movies[:-1])
            train_genres.append(train_genre[:-1])
            targets_train.append(train_movies[-1])

        if len(val_movies) > SEQ_LENGTH:
            val_data.append(val_movies[:-1])
            val_genres.append(val_genre[:-1])
            targets_val.append(val_movies[-1])

        if len(test_movies) > SEQ_LENGTH:
            test_data.append(test_movies[:-1])
            test_genres.append(test_genre[:-1])
            targets_test.append(test_movies[-1])

    return train_data, val_data, test_data, train_genres, val_genres, test_genres, targets_train, targets_val, targets_test


def build_interactions(data):
    """Build variable-length user interaction sequences."""
    user_interactions, genre_interactions, targets = [], [], []
    users = data['userId'].unique()

    for user in tqdm(users, desc='Processing Users'):
        user_data = data[data['userId'] == user].sort_values('timestamp')
        movies_list = user_data['movieId'].tolist()
        genres_list = user_data['genres'].tolist()

        # Generate variable sequence lengths (between 5-10)
        if len(movies_list) > SEQ_LENGTH:
            seq_length = random.randint(5, SEQ_LENGTH)
            movies_list = movies_list[-(seq_length + 1):]  # Last N interactions
            genres_list = genres_list[-(seq_length + 1):]

        if len(movies_list) < SEQ_LENGTH + 1:
            continue  # Skip short sequences

        user_interactions.append(movies_list[:-1])
        genre_interactions.append(genres_list[:-1])
        targets.append(movies_list[-1])

    return user_interactions, genre_interactions, targets


def split_and_save(train_data, val_data, test_data, train_genres, val_genres, test_genres, train_targets, val_targets,
                   test_targets, output_prefix):
    """Save train, validation, and test datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pd.DataFrame({'seq': train_data, 'len_seq': [len(x) for x in train_data], 'target': train_targets}).to_csv(
        os.path.join(OUTPUT_DIR, f'{output_prefix}_train.df'), index=False)
    pd.DataFrame({'seq': val_data, 'len_seq': [len(x) for x in val_data], 'target': val_targets}).to_csv(
        os.path.join(OUTPUT_DIR, f'{output_prefix}_val.df'), index=False)
    pd.DataFrame({'seq': test_data, 'len_seq': [len(x) for x in test_data], 'target': test_targets}).to_csv(
        os.path.join(OUTPUT_DIR, f'{output_prefix}_test.df'), index=False)

    pd.DataFrame({'genres': train_genres}).to_csv(os.path.join(OUTPUT_DIR, f'{output_prefix}_train_g.df'), index=False)
    pd.DataFrame({'genres': val_genres}).to_csv(os.path.join(OUTPUT_DIR, f'{output_prefix}_val_g.df'), index=False)
    pd.DataFrame({'genres': test_genres}).to_csv(os.path.join(OUTPUT_DIR, f'{output_prefix}_test_g.df'), index=False)


if __name__ == "__main__":
    download_and_extract()
    data = load_data()

    # Apply filters to prevent overfitting
    filtered_data = filter_users(data)
    filtered_data = filter_popular_movies(filtered_data)  # Remove top 5% most popular movies

    # Prepare data for training using improved time-based splitting
    train_data, val_data, test_data, train_genres, val_genres, test_genres, train_targets, val_targets, test_targets = time_based_split(
        filtered_data)
    # Save processed data
    split_and_save(train_data, val_data, test_data, train_genres, val_genres, test_genres, train_targets, val_targets,
                   test_targets, "movie_data")

    print("Processing complete! All datasets saved in 'Amir' folder.")
