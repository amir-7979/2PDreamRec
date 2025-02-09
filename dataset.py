import os
import zipfile
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
import random

# -----------------------------
# Constants and Directories
# -----------------------------
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "movielens_1m"
EXTRACTED_DIR = os.path.join(DATA_DIR, "ml-1m")
OUTPUT_DIR = "data"  # Directory for merged nested fold data

# Define sequence lengths:
RAW_SEQ_LENGTH = 11  # Total number of interactions extracted (including target)
TRAIN_SEQ_LENGTH = RAW_SEQ_LENGTH - 1  # The training sequence length (i.e. without target)

N_FOLDS = 10  # Number of outer folds for nested CV


# -----------------------------
# Download and Extraction
# -----------------------------
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


# -----------------------------
# Data Loading and Filtering
# -----------------------------
def load_data():
    """Load ratings and movie metadata and merge them."""
    ratings_path = os.path.join(EXTRACTED_DIR, "ratings.dat")
    movies_path = os.path.join(EXTRACTED_DIR, "movies.dat")
    ratings = pd.read_csv(ratings_path, sep="::", engine="python",
                          names=["userId", "movieId", "rating", "timestamp"],
                          encoding="ISO-8859-1")
    movies = pd.read_csv(movies_path, sep="::", engine="python",
                         names=["movieId", "title", "genres"],
                         encoding="ISO-8859-1")
    return pd.merge(ratings, movies, on="movieId")


def filter_users(data, min_interactions=5, max_interactions=300, min_high_rating=5):
    """
    Keep only users who:
      1. Have total interactions between min_interactions and max_interactions, and
      2. Have at least min_high_rating interactions with a rating of 4 or above.
    """
    # First, filter by total interactions.
    user_counts = data['userId'].value_counts()
    valid_users_total = user_counts[(user_counts >= min_interactions) & (user_counts <= max_interactions)].index
    data = data[data['userId'].isin(valid_users_total)]

    # Now, for the remaining data, count per user how many interactions have rating >= 4.
    high_rating_counts = data[data['rating'] >= 4].groupby('userId').size()
    valid_high_rating_users = high_rating_counts[high_rating_counts >= min_high_rating].index

    # Finally, keep only those users who satisfy both criteria.
    return data[data['userId'].isin(valid_high_rating_users)]


def filter_popular_movies(data, threshold=0.01):
    """Remove the top percentage of popular movies."""
    movie_counts = data['movieId'].value_counts()
    top_movies = movie_counts.nlargest(int(len(movie_counts) * threshold)).index
    return data[~data['movieId'].isin(top_movies)]


# -----------------------------
# Build Interactions
# -----------------------------
def build_interactions(data):
    """
    For each user, build:
      - movie_interactions: list of movie IDs (first TRAIN_SEQ_LENGTH items from the raw list)
      - genre_interactions: list of genre IDs corresponding to the movie interactions
      - movie_targets: the target movie (the last item in the raw list)
      - genre_targets: the target genre (first genre of the target movie)
    Also build a mapping (genre2id) from genre string to integer.
    """
    movie_interactions = []
    genre_interactions = []
    movie_targets = []
    genre_targets = []
    genre2id = {}
    current_genre_id = 0

    users = data['userId'].unique()
    for user in tqdm(users, desc='Processing Users'):
        user_data = data[data['userId'] == user].sort_values('timestamp')
        movies_list = user_data['movieId'].tolist()
        genres_list = user_data['genres'].tolist()  # e.g., "Comedy|Romance"
        if len(movies_list) < RAW_SEQ_LENGTH:
            continue  # Skip users with too few interactions
        # Use only the last RAW_SEQ_LENGTH interactions
        movies_list = movies_list[-RAW_SEQ_LENGTH:]
        genres_list = genres_list[-RAW_SEQ_LENGTH:]
        # The training sequence is the first TRAIN_SEQ_LENGTH items, and the target is the last item.
        movie_interactions.append(movies_list[:-1])
        movie_targets.append(movies_list[-1])
        # Build genre sequence: for each movie in the training sequence, take its first genre.
        genre_seq = []
        for g in genres_list[:-1]:
            first_genre = g.split("|")[0] if isinstance(g, str) else str(g)
            if first_genre not in genre2id:
                genre2id[first_genre] = current_genre_id
                current_genre_id += 1
            genre_seq.append(genre2id[first_genre])
        genre_interactions.append(genre_seq)
        # For target, take the first genre of the target movie.
        target_genre_str = genres_list[-1].split("|")[0] if isinstance(genres_list[-1], str) else str(genres_list[-1])
        if target_genre_str not in genre2id:
            genre2id[target_genre_str] = current_genre_id
            current_genre_id += 1
        genre_targets.append(genre2id[target_genre_str])
    return movie_interactions, genre_interactions, movie_targets, genre_targets, genre2id


# -----------------------------
# Save Merged Nested Fold Data
# -----------------------------
def save_nested_fold_merged_data(movies, movie_targets, genres, genre_targets, fold_no, split_type):
    """
    Save merged nested fold data into a single CSV file with columns:
      - movie_seq, movie_len, movie_target, genre_seq, genre_len, genre_target.
    Here, movie_len and genre_len are set to TRAIN_SEQ_LENGTH.
    split_type should be 'train', 'val', or 'test'.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_dict = {
        "movie_seq": movies,
        "movie_len": [TRAIN_SEQ_LENGTH for _ in movies],
        "movie_target": movie_targets,
        "genre_seq": genres,
        "genre_len": [TRAIN_SEQ_LENGTH for _ in genres],
        "genre_target": genre_targets
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(OUTPUT_DIR, f"{split_type}_fold{fold_no}.df"), index=False)


# -----------------------------
# Main Nested Splitting Procedure
# -----------------------------
if __name__ == "__main__":
    download_and_extract()
    data = load_data()
    print("Data loaded:", data.shape)

    # Apply filters
    filtered_data = filter_users(data)
    #filtered_data = filter_popular_movies(filtered_data)
    print("Filtered data:", filtered_data.shape)

    # Build interactions for movies and genres
    movie_interactions, genre_interactions, movie_targets, genre_targets, genre2id = build_interactions(filtered_data)
    print("Number of interaction sequences:", len(movie_interactions))
    print("Number of unique genres:", len(genre2id))

    # Outer KFold splitting for nested cross-validation
    kf_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_no = 1

    for outer_train_index, outer_test_index in kf_outer.split(movie_interactions):
        print(f"Processing Outer Fold {fold_no}")
        # Outer test data remains the same.
        movies_test = [movie_interactions[i] for i in outer_test_index]
        movie_targets_test = [movie_targets[i] for i in outer_test_index]
        genres_test = [genre_interactions[i] for i in outer_test_index]
        genre_targets_test = [genre_targets[i] for i in outer_test_index]

        # Split outer training data to obtain a validation subset (for monitoring)
        # However, we now use the entire outer_train_index as the training set.
        inner_train_index, inner_val_index = train_test_split(outer_train_index, test_size=0.1, random_state=42)

        # Training set (union of inner_train and inner_val)
        movies_train = [movie_interactions[i] for i in outer_train_index]
        movie_targets_train = [movie_targets[i] for i in outer_train_index]
        genres_train = [genre_interactions[i] for i in outer_train_index]
        genre_targets_train = [genre_targets[i] for i in outer_train_index]

        # Validation set (subset for monitoring; note that these examples are also in training)
        movies_val = [movie_interactions[i] for i in inner_val_index]
        movie_targets_val = [movie_targets[i] for i in inner_val_index]
        genres_val = [genre_interactions[i] for i in inner_val_index]
        genre_targets_val = [genre_targets[i] for i in inner_val_index]

        # Save merged nested fold data (for each fold and each split)
        save_nested_fold_merged_data(movies_train, movie_targets_train, genres_train, genre_targets_train, fold_no,
                                     "train")
        save_nested_fold_merged_data(movies_val, movie_targets_val, genres_val, genre_targets_val, fold_no, "val")
        save_nested_fold_merged_data(movies_test, movie_targets_test, genres_test, genre_targets_test, fold_no, "test")

        print(f"Saved merged fold {fold_no} files.")
        fold_no += 1

    # -----------------------------
    # Compute and Save Statistics
    # -----------------------------
    # Compute unique movies (from both the sequences and the targets)
    unique_movies = set()
    for seq in movie_interactions:
        unique_movies.update(seq)
    unique_movies.update(movie_targets)
    movie_count = len(unique_movies)

    num_users = len(movie_interactions)
    num_genres = len(genre2id)
    avg_movie_seq_length = np.mean([len(seq) for seq in movie_interactions])
    avg_genre_seq_length = np.mean([len(seq) for seq in genre_interactions])

    statics_dict = {
        "num_users": num_users,
        "num_movies": movie_count,
        "num_genres": num_genres,
        "train_seq_length": TRAIN_SEQ_LENGTH,
        "raw_seq_length": RAW_SEQ_LENGTH,
        "avg_movie_seq_length": avg_movie_seq_length,
        "avg_genre_seq_length": avg_genre_seq_length
    }
    statics_df = pd.DataFrame(list(statics_dict.items()), columns=["statistic", "value"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    statics_df.to_csv(os.path.join(OUTPUT_DIR, "statics.csv"), index=False)
    print("Saved statics.csv with dataset information.")

    pd.DataFrame(list(genre2id.items()), columns=['genre', 'id']).to_csv(os.path.join(OUTPUT_DIR, "genre_mapping.csv"),
                                                                         index=False)

    print("Nested 10-Fold dataset preparation complete!")
