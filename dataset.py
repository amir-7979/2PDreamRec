import os
import zipfile
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
import random
import torch
import json

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
# Helper Function: Padding by Last Sequence
# -----------------------------
def pad_by_last_sequence(seq, desired_length):
    current_length = len(seq)
    if current_length >= desired_length:
        return seq[:desired_length]
    missing = desired_length - current_length
    if missing <= current_length:
        pad = seq[-missing:]
    else:
        pad = []
        while len(pad) < missing:
            pad.extend(seq[-min(missing - len(pad), current_length):])
    return seq + pad

# -----------------------------
# Download and Extraction
# -----------------------------
def download_and_extract():
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
# Helper Functions for Reindexing
# -----------------------------
def reindex_movies(movie_df):
    unique_ids = sorted(movie_df['movieId'].unique())
    new_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
    movie_df['movieId'] = movie_df['movieId'].map(new_mapping)
    return movie_df, new_mapping

def reindex_merged_data(df):
    unique_ids = sorted(df['movieId'].unique())
    new_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
    df['movieId'] = df['movieId'].map(new_mapping)
    return df, new_mapping

# -----------------------------
# Data Loading and Filtering
# -----------------------------
def load_data():
    """Load ratings and movie metadata, reindex movie IDs, and merge them."""
    ratings_path = os.path.join(EXTRACTED_DIR, "ratings.dat")
    movies_path = os.path.join(EXTRACTED_DIR, "movies.dat")
    ratings = pd.read_csv(ratings_path, sep="::", engine="python",
                          names=["userId", "movieId", "rating", "timestamp"],
                          encoding="ISO-8859-1")
    movies = pd.read_csv(movies_path, sep="::", engine="python",
                         names=["movieId", "title", "genres"],
                         encoding="ISO-8859-1")
    # Reindex movies and update ratings accordingly.
    movies, mapping = reindex_movies(movies)
    ratings['movieId'] = ratings['movieId'].map(mapping)
    merged = pd.merge(ratings, movies, on="movieId")
    # Alternatively, reindex the merged DataFrame (if needed):
    # merged, _ = reindex_merged_data(merged)
    return merged






import os
import pandas as pd
import json

def build_genre_movie_mapping(movies_path, output_path):
    movies = pd.read_csv(movies_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="ISO-8859-1")
    genre_movie_mapping = {}
    genre2id = {}
    current_genre_id = 0
    
    for _, row in movies.iterrows():
        movie_id = row["movieId"]
        genres = row["genres"].split("|")
        for genre in genres:
            if genre not in genre2id:
                genre2id[genre] = current_genre_id
                genre_movie_mapping[current_genre_id] = []
                current_genre_id += 1
            genre_id = genre2id[genre]
            genre_movie_mapping[int(genre_id)].append(movie_id)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(genre_movie_mapping, f, indent=4)






def filter_users(data, min_interactions=5, max_interactions=4000, min_high_rating=5):
    user_counts = data['userId'].value_counts()
    valid_users_total = user_counts[(user_counts >= min_interactions) & (user_counts <= max_interactions)].index
    data = data[data['userId'].isin(valid_users_total)]
    high_rating_counts = data[data['rating'] >= 4].groupby('userId').size()
    valid_high_rating_users = high_rating_counts[high_rating_counts >= min_high_rating].index
    return data[data['userId'].isin(valid_high_rating_users)]

# -----------------------------
# Build Interactions
# -----------------------------
def build_interactions(data):
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
            movies_list = pad_by_last_sequence(movies_list, RAW_SEQ_LENGTH)
            genres_list = pad_by_last_sequence(genres_list, RAW_SEQ_LENGTH)
        else:
            movies_list = movies_list[-RAW_SEQ_LENGTH:]
            genres_list = genres_list[-RAW_SEQ_LENGTH:]
        movie_interactions.append(movies_list[:-1])
        movie_targets.append(movies_list[-1])
        genre_seq = []
        for g in genres_list[:-1]:
            first_genre = g.split("|")[0] if isinstance(g, str) else str(g)
            if first_genre not in genre2id:
                genre2id[first_genre] = current_genre_id
                current_genre_id += 1
            genre_seq.append(genre2id[first_genre])
        genre_interactions.append(genre_seq)
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


def reindex_filtered_data(df):
    # Get all unique movie IDs in the filtered data
    unique_ids = sorted(df['movieId'].unique())
    # Build a mapping from old movieId to new movieId (starting at 1)
    new_mapping = {old: new for new, old in enumerate(unique_ids, start=1)}
    df['movieId'] = df['movieId'].map(new_mapping)
    return df, new_mapping


# -----------------------------
# Main Nested Splitting Procedure (No Validation)
# -----------------------------
if __name__ == "__main__":
    download_and_extract()
    data = load_data()
    print("Data loaded:", data.shape)
    filtered_data = filter_users(data)
    filtered_data, movie_mapping = reindex_filtered_data(filtered_data)

    # Optionally, you can filter popular movies:
    # filtered_data = filter_popular_movies(filtered_data)
    print("Filtered data:", filtered_data.shape)
    movie_interactions, genre_interactions, movie_targets, genre_targets, genre2id = build_interactions(filtered_data)
    print("Number of interaction sequences:", len(movie_interactions))
    print("Number of unique genres:", len(genre2id))
    kf_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    for outer_train_index, outer_test_index in kf_outer.split(movie_interactions):
        movies_train = [movie_interactions[i] for i in outer_train_index]
        movie_targets_train = [movie_targets[i] for i in outer_train_index]
        genres_train = [genre_interactions[i] for i in outer_train_index]
        genre_targets_train = [genre_targets[i] for i in outer_train_index]
        movies_test = [movie_interactions[i] for i in outer_test_index]
        movie_targets_test = [movie_targets[i] for i in outer_test_index]
        genres_test = [genre_interactions[i] for i in outer_test_index]
        genre_targets_test = [genre_targets[i] for i in outer_test_index]
        save_nested_fold_merged_data(movies_train, movie_targets_train, genres_train, genre_targets_train, fold_no, "train")
        save_nested_fold_merged_data(movies_test, movie_targets_test, genres_test, genre_targets_test, fold_no, "test")
        fold_no += 1
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
        "num_movies": movie_count,   # Now computed from the reindexed interactions
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
    pd.DataFrame(list(genre2id.items()), columns=['genre', 'id']).to_csv(
        os.path.join(OUTPUT_DIR, "genre_mapping.csv"), index=False
    )
    movies_path = os.path.join(EXTRACTED_DIR, "movies.dat")
    mapping_output_path = os.path.join(OUTPUT_DIR, "movie_to_genre_mapping.csv")
    print("Nested 10-Fold dataset preparation complete!")
    output_path = os.path.join(OUTPUT_DIR, "genre_movie_mapping.json")
    build_genre_movie_mapping(movies_path, output_path)
