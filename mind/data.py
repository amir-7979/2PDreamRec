import os
import zipfile
import requests
import pandas as pd, numpy as np
from tqdm import tqdm
import json

MIND_URLS = {
    "train": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip",
    "dev": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_dev.zip",
    "test": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_test.zip"
}

DATA_DIR = "data"
OUTPUT_DIR = "data"
RAW_SEQ_LENGTH = 11
TRAIN_SEQ_LENGTH = RAW_SEQ_LENGTH - 1


def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in MIND_URLS.items():
        zip_path = os.path.join(DATA_DIR, f"MIND_{name}.zip")
        extract_dir = os.path.join(DATA_DIR, f"MIND_{name}")
        
        if not os.path.exists(extract_dir):
            if not os.path.exists(zip_path):
                print(f"Downloading {name} dataset from {url} ...")
                try:
                    r = requests.get(url, stream=True, timeout=20)
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    block_size = 1024

                    if total_size == 0:
                        print(f"Warning: Cannot determine file size for {name}. Downloading without progress bar.")
                        with open(zip_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=block_size):
                                if chunk:
                                    f.write(chunk)
                    else:
                        with open(zip_path, "wb") as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {name}") as pbar:
                            for chunk in r.iter_content(chunk_size=block_size):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    print(f"Downloaded {name} dataset.")
                except Exception as e:
                    print(f"❌ Error downloading {name} dataset: {e}")
                    continue
            else:
                print(f"{name}.zip already exists, skipping download.")
            
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Extracted {name} dataset to {extract_dir}.")
            except Exception as e:
                print(f"❌ Error extracting {name}.zip: {e}")
        else:
            print(f"{extract_dir} already exists, skipping extraction.")


def load_behavior_news(split='train'):
    print(f"\nLoading {split} behaviors and news...")
    behavior_path = os.path.join(DATA_DIR, f"MIND_{split}", "behaviors.tsv")
    news_path = os.path.join(DATA_DIR, f"MIND_{split}", "news.tsv")
    behaviors = pd.read_csv(behavior_path, sep="\t", header=None,
                            names=["ImpressionID", "UserID", "Time", "History", "Impressions"])
    news = pd.read_csv(news_path, sep="\t", header=None,
                       names=["NewsID", "Category", "SubCategory", "Title", "Abstract", "URL", "TitleEntities", "AbstractEntities"])
    print(f"Loaded {len(behaviors)} behaviors, {len(news)} news items.")
    return behaviors, news


def build_genre_mapping(news_df):
    print("Building genre (category) to ID mapping...")
    genres = sorted(news_df['Category'].dropna().unique())
    genre2id = {genre: idx for idx, genre in enumerate(genres)}
    print(f"Found {len(genre2id)} unique genres.")
    return genre2id

def process_split(split='train', min_interactions=5):
    print(f"\n=== Processing split: {split} ===")
    behaviors, news = load_behavior_news(split)
    genre2id = build_genre_mapping(news)
    news2genre = dict(zip(news['NewsID'], news['Category']))

    # Build news ID to int mapping (like movieId)
    unique_news_ids = news['NewsID'].unique()
    news2id = {nid: idx for idx, nid in enumerate(unique_news_ids, start=1)}

    # Build user histories
    user_histories = {}
    for _, row in behaviors.iterrows():
        uid = row['UserID']
        if pd.isna(row['History']):
            continue
        history = row['History'].split()
        user_histories.setdefault(uid, []).extend(history)

    print(f"Users before filtering: {len(user_histories)}")
    user_histories = {uid: hist for uid, hist in user_histories.items() if len(hist) >= RAW_SEQ_LENGTH}
    print(f"Users after filtering (≥{RAW_SEQ_LENGTH}): {len(user_histories)}")

    movie_seq, movie_target = [], []
    genre_seq, genre_target = [], []

    for uid, history in tqdm(user_histories.items(), desc=f"Processing users in {split}"):
        history = history[-RAW_SEQ_LENGTH:]
        hist_items, target_item = history[:-1], history[-1]

        # Convert to integer IDs
        if any(nid not in news2id for nid in history):
            continue

        hist_ids = [news2id[nid] for nid in hist_items]
        target_id = news2id.get(target_item)

        genre_items = [genre2id.get(news2genre.get(nid, ""), -1) for nid in hist_items]
        target_genre = genre2id.get(news2genre.get(target_item, ""), -1)

        if -1 in genre_items or target_genre == -1:
            continue

        movie_seq.append(hist_ids)
        movie_target.append(target_id)
        genre_seq.append(genre_items)
        genre_target.append(target_genre)

    print(f"Final sequences prepared: {len(movie_seq)}")

    df = pd.DataFrame({
        "movie_seq": movie_seq,
        "movie_len": [len(seq) for seq in movie_seq],
        "movie_target": movie_target,
        "genre_seq": genre_seq,
        "genre_len": [len(seq) for seq in genre_seq],
        "genre_target": genre_target
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{split}.df")
    df.to_csv(out_path, index=False)
    print(f"Saved {split}.df to {out_path}")

    if split == "train":
        # Save genre and news mappings only once (on train split)
        pd.DataFrame(list(genre2id.items()), columns=["genre", "id"]).to_csv(
            os.path.join(OUTPUT_DIR, "genre_mapping.csv"), index=False
        )
        pd.DataFrame(list(news2id.items()), columns=["news_id", "movie_id"]).to_csv(
            os.path.join(OUTPUT_DIR, "news_id_mapping.csv"), index=False
        )

        stats = {
            "num_users": len(user_histories),
            "num_news": len(news),
            "num_genres": len(genre2id),
            "train_seq_length": TRAIN_SEQ_LENGTH,
            "raw_seq_length": RAW_SEQ_LENGTH,
            "avg_movie_seq_length": np.mean([len(seq) for seq in movie_seq])
        }
        pd.DataFrame(list(stats.items()), columns=["statistic", "value"]).to_csv(
            os.path.join(OUTPUT_DIR, "statics.csv"), index=False
        )
        print("Saved genre_mapping.csv, news_id_mapping.csv, and statics.csv.")


if __name__ == "__main__":
    print("===== STARTING MIND DATA PREPARATION =====")
    download_and_extract()
    process_split("train")
    process_split("dev")
    process_split("test")
    print("===== MIND DATA PREPARATION COMPLETE =====")
