import os
import zipfile
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
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

# Global settings for the original experiment
RAW_SEQ_LENGTH = 11  # Total number of interactions extracted (including target)
TRAIN_SEQ_LENGTH = RAW_SEQ_LENGTH - 1  # Training sequence length
N_FOLDS = 10  # Number of outer folds for nested CV

# -----------------------------
# Helper Function: Padding by Last Item
# -----------------------------
def pad_by_last_sequence(seq, desired_length):
    current_length = len(seq)
    if current_length >= desired_length:
        return seq[:desired_length]
    # Copy the last item until the list reaches the desired length.
    pad = [seq[-1]] * (desired_length - current_length)
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
# Data Loading and Merging
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
    return merged

def build_genre_movie_mapping(movies_path, output_path):
    movies = pd.read_csv(movies_path, sep="::", engine="python", names=["movieId", "title", "genres"],
                         encoding="ISO-8859-1")
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

# -----------------------------
# Filtering Functions
# -----------------------------
def filter_users_original(data):
    """
    For the original experiment:
      - Only consider interactions with rating >= 4.
      - Only keep users with at least 5 interactions (and at most 4000).
    """
    data_filtered = data[data['rating'] >= 4]
    user_counts = data_filtered['userId'].value_counts()
    valid_users = user_counts[(user_counts >= 5) & (user_counts <= 4000)].index
    return data_filtered[data_filtered['userId'].isin(valid_users)]

def filter_users_exp(data, min_interactions, rating_threshold, max_interactions=4000):
    """
    Generic filter for experiments.
      - Keeps interactions with rating >= rating_threshold.
      - Keeps users with at least min_interactions (and at most max_interactions).
    """
    data_filtered = data[data['rating'] >= rating_threshold]
    user_counts = data_filtered['userId'].value_counts()
    valid_users = user_counts[(user_counts >= min_interactions) & (user_counts <= max_interactions)].index
    return data_filtered[data_filtered['userId'].isin(valid_users)]


# -----------------------------
# Build Interactions
# -----------------------------
def build_interactions_original(data):
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


def build_interactions_experiment(data, raw_seq_length):
    """
    Builds movie and genre interaction sequences for each user using the given raw_seq_length.
    Returns:
      movie_interactions, genre_interactions, movie_targets, genre_targets, genre2id mapping.
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
        if len(movies_list) < raw_seq_length:
            movies_list = pad_by_last_sequence(movies_list, raw_seq_length)
            genres_list = pad_by_last_sequence(genres_list, raw_seq_length)
        else:
            movies_list = movies_list[-raw_seq_length:]
            genres_list = genres_list[-raw_seq_length:]
        # Build training sequence (all but the last interaction) and target.
        training_seq = movies_list[:-1]
        target = movies_list[-1]
        # Check if training sequence has the expected length.
        if len(training_seq) != raw_seq_length - 1:
            print(f"User {user} has training sequence length {len(training_seq)} (expected {raw_seq_length - 1}).")
        movie_interactions.append(training_seq)
        movie_targets.append(target)
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

# A convenience function for the original experiment that uses the global RAW_SEQ_LENGTH (11)
def build_interactions(data):
    return build_interactions_experiment(data, RAW_SEQ_LENGTH)

# -----------------------------
# Saving Function
# -----------------------------
def save_nested_fold_merged_data(movies, movie_targets, genres, genre_targets, fold_no, split_type):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Here, we use the actual sequence lengths rather than a fixed TRAIN_SEQ_LENGTH,
    # so that it works correctly for both the original experiment and the experimental folds.
    data_dict = {
        "movie_seq": movies,
        "movie_len": [len(seq) for seq in movies],
        "movie_target": movie_targets,
        "genre_seq": genres,
        "genre_len": [len(seq) for seq in genres],
        "genre_target": genre_targets
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(OUTPUT_DIR, f"{split_type}_fold{fold_no}.df"), index=False)

def reindex_filtered_data(df):
    unique_ids = sorted(df['movieId'].unique())
    new_mapping = {old: new for new, old in enumerate(unique_ids, start=1)}
    df['movieId'] = df['movieId'].map(new_mapping)
    return df, new_mapping

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Download, extract, and load data
    download_and_extract()
    data = load_data()
    print("Data loaded:", data.shape)
    
    # Reindex the data without filtering
    data, movie_mapping = reindex_filtered_data(data)
    print("Reindexed data:", data.shape)
    
    # -----------------------------
    # Original 10-Fold Nested CV (Original Experiment)
    # Settings: RAW_SEQ_LENGTH=11, min_interactions=5, rating >= 4
    unique_users = data['userId'].unique()
    kf_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    for train_idx, test_idx in kf_outer.split(unique_users):
        train_users = unique_users[train_idx]
        test_users = unique_users[test_idx]
    
        train_data = data[data['userId'].isin(train_users)]
        test_data = data[data['userId'].isin(test_users)]
    
        # Filter training data for original experiment
        filtered_train_data = filter_users_original(train_data)
    
        # Build interactions using original settings (RAW_SEQ_LENGTH=11)
        movie_interactions_train, genre_interactions_train, movie_targets_train, genre_targets_train, genre2id_train = build_interactions_original(filtered_train_data)
        movie_interactions_test, genre_interactions_test, movie_targets_test, genre_targets_test, genre2id_test = build_interactions_original(test_data)
    
        save_nested_fold_merged_data(movie_interactions_train, movie_targets_train, genre_interactions_train, genre_targets_train, fold_no, "train")
        save_nested_fold_merged_data(movie_interactions_test, movie_targets_test, genre_interactions_test, genre_targets_test, fold_no, "test")
    
        fold_no += 1
    
    # -----------------------------
    # Experimental Fold p1
    # Settings: RAW_SEQ_LENGTH=4, min_interactions=1, rating >= 0
    print("Creating experimental fold 'p1' ...")
    user_interaction_counts = data['userId'].value_counts()
    sorted_users = user_interaction_counts.sort_values(ascending=False).index.tolist()
    cutoff_index = int(len(sorted_users) * 0.8)
    train_users_p1 = sorted_users[:cutoff_index]
    test_users_p1 = sorted_users[cutoff_index:]
    
    train_data_p1 = data[data['userId'].isin(train_users_p1)]
    test_data_p1 = data[data['userId'].isin(test_users_p1)]
    
    filtered_train_data_p1 = filter_users_exp(train_data_p1, min_interactions=1, rating_threshold=0)
    movie_interactions_train_p1, genre_interactions_train_p1, movie_targets_train_p1, genre_targets_train_p1, genre2id_train_p1 = build_interactions_experiment(filtered_train_data_p1, 4)
    movie_interactions_test_p1, genre_interactions_test_p1, movie_targets_test_p1, genre_targets_test_p1, genre2id_test_p1 = build_interactions_experiment(test_data_p1, 4)
    
    save_nested_fold_merged_data(movie_interactions_train_p1, movie_targets_train_p1, genre_interactions_train_p1, genre_targets_train_p1, "p1", "train")
    save_nested_fold_merged_data(movie_interactions_test_p1, movie_targets_test_p1, genre_interactions_test_p1, genre_targets_test_p1, "p1", "test")
    
    # -----------------------------
    # Experimental Fold p2
    # Settings: RAW_SEQ_LENGTH=11, min_interactions=1, rating >= 4
    print("Creating experimental fold 'p2' ...")
    # Reuse the same user split as in p1
    train_data_p2 = data[data['userId'].isin(train_users_p1)]
    test_data_p2 = data[data['userId'].isin(test_users_p1)]
    
    filtered_train_data_p2 = filter_users_exp(train_data_p2, min_interactions=1, rating_threshold=4)
    movie_interactions_train_p2, genre_interactions_train_p2, movie_targets_train_p2, genre_targets_train_p2, genre2id_train_p2 = build_interactions_experiment(filtered_train_data_p2, 11)
    movie_interactions_test_p2, genre_interactions_test_p2, movie_targets_test_p2, genre_targets_test_p2, genre2id_test_p2 = build_interactions_experiment(test_data_p2, 11)
    
    save_nested_fold_merged_data(movie_interactions_train_p2, movie_targets_train_p2, genre_interactions_train_p2, genre_targets_train_p2, "p2", "train")
    save_nested_fold_merged_data(movie_interactions_test_p2, movie_targets_test_p2, genre_interactions_test_p2, genre_targets_test_p2, "p2", "test")
    
    # -----------------------------
    # Overall Statistics and Saving Genre Mapping
    unique_movies = set()
    for seq in movie_interactions_train + movie_interactions_test:
        unique_movies.update(seq)
    unique_movies.update(movie_targets_train + movie_targets_test)
    movie_count = len(unique_movies)
    num_users = len(data['userId'].unique())
    num_genres = len(genre2id_train)
    avg_movie_seq_length = np.mean([len(seq) for seq in (movie_interactions_train + movie_interactions_test)])
    avg_genre_seq_length = np.mean([len(seq) for seq in (genre_interactions_train + genre_interactions_test)])
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
    
    # Save genre mapping (using the training mapping from the original experiment)
    pd.DataFrame(list(genre2id_train.items()), columns=['genre', 'id']).to_csv(
        os.path.join(OUTPUT_DIR, "genre_mapping.csv"), index=False
    )
    movies_path = os.path.join(EXTRACTED_DIR, "movies.dat")
    mapping_output_path = os.path.join(OUTPUT_DIR, "movie_to_genre_mapping.csv")
    print("Nested 10-Fold dataset preparation complete!")
    output_path = os.path.join(OUTPUT_DIR, "genre_movie_mapping.json")
    build_genre_movie_mapping(movies_path, output_path)
