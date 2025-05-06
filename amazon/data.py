#!/usr/bin/env python3
import os, gzip, json, requests, urllib3, random
import pandas as pd, numpy as np
from tqdm import tqdm

# ───── CONFIG ───────────────────────────────────────────────────────────
REVIEWS_JSONL_URL   = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz"
METADATA_JSONL_URL  = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Movies_and_TV.jsonl.gz"

DATA_DIR           = "data"
OUTPUT_DIR         = "data"

RAW_SEQ_LEN        = 11       # sequence length including target
MIN_INTERACTIONS   = 6        # minimum # of ≥4★ reviews per user
RATING_THRESHOLD   = 4.0      # threshold
# ────────────────────────────────────────────────────────────────────────

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
random.seed(42)
np.random.seed(42)

# your 200+ StudioBinder genres list
STANDARD_GENRES = [
    "Action","Adventure","Animation","Anime","Biography","Children's","Comedy","Crime","Dance",
    "Disaster","Documentary","Drama","Family","Fantasy","Film Noir","Historical","Holiday","Horror",
    "Legal Drama","Martial Arts","Medical Drama","Mockumentary","Musical","Mystery","Parody",
    "Period Drama","Political Drama","Political Thriller","Psychological Thriller","Reality TV",
    "Religious","Romance","Romantic Comedy","Romantic Drama","Satire","Science Fiction","Silent Film",
    "Slice of Life","Soap Opera","Sports","Spy","Stand-Up Comedy","Superhero","Suspense","Talk Show",
    "Teen Drama","Thriller","War","Western","Absurdist","Action-Comedy","Anthology","Apocalyptic",
    "Art House","B-Movie","Black Comedy","Blaxploitation","Body Horror","Buddy Comedy","Coming-of-Age",
    "Courtroom Drama","Cyberpunk","Dance Film","Detective","Docudrama","Dramedy","Eco-Thriller",
    "Erotic Thriller","Experimental","Fairy Tale","Fantasy Comedy","Found Footage","Giallo",
    "Girls with Guns","Gothic Horror","Heist","High Fantasy","Historical Epic","Home Improvement",
    "Hood Film","Hybrid Genre","Infotainment","Interactive","Jidaigeki","LGBTQ+","Magical Realism",
    "Mecha","Mockbuster","Monster Movie","Mumblecore","Musical Comedy","Mythological","Nature Documentary",
    "Neo-Noir","Paranormal Romance","Psychological Drama","Road Movie","Space Opera","Zombie",
    "Anthropological Drama","Art Drama","Biblical Drama","Business Drama","Campus Drama",
    "Detective Drama","Disaster Drama","Family Saga","Gangster Drama","Historical Fiction Drama",
    "Industrial Drama","Legal Thriller","Medical Thriller","Military Drama","Mythological Drama",
    "Paranormal Drama","Philosophical Drama","Political Satire","Comedy of Manners","Cringe Comedy",
    "Deadpan Comedy","Farce","Improvisational Comedy","Sketch Comedy","Slapstick","Surreal Comedy",
    "Teen Comedy","Workplace Comedy","Adventure Reality","Celebrity Reality","Competition Reality",
    "Cooking Competition","Dating Show","Docu-Reality","Extreme Sports Reality","Fashion Reality",
    "Home Renovation Show","Lifestyle Reality","Makeover Show","Music Competition","Paranormal Reality",
    "Survival Show","Talent Show","Travel Reality","True Crime Reality","Variety Show","Weight Loss Show",
    "Wildlife Reality","Biography Documentary","Crime Documentary","Cultural Documentary",
    "Environmental Documentary","Historical Documentary","Investigative Journalism","Music Documentary",
    "Nature Documentary","Political Documentary","Science Documentary","Social Issues Documentary",
    "Sports Documentary","Technology Documentary","Travel Documentary","True Crime Documentary",
    "War Documentary","Wildlife Documentary","World Affairs Documentary","Youth Issues Documentary",
    "Zoological Documentary","Animated Sitcom","Apocalyptic Fiction","Dystopian Fiction",
    "Fantasy Adventure","Historical Fantasy","Legal Procedural","Martial Arts Series","Mecha Anime",
    "Mystery Thriller","Period Piece","Police Procedural","Post-Apocalyptic Fiction",
    "Supernatural Thriller","Time Travel Series","Action Series","Adventure Series","Animated Series",
    "Children","Cooking Show","Daytime","Dramatic","Educational","Factual","Game Show","Infomercials",
    "Late Night","Music Television","News Programming","Religious Programming","Other"
]

def download_if_missing(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"📂 {os.path.basename(dest)} exists, skipping")
        return
    print(f"⬇️  Downloading {os.path.basename(dest)} …")
    r = requests.get(url, stream=True, verify=False, timeout=60); r.raise_for_status()
    with open(dest,"wb") as f:
        for chunk in r.iter_content(1024*64):
            f.write(chunk)
    print("✅ Download complete.")

def load_metadata(path):
    print("📥 Loading metadata…")
    rows=[]; opener=gzip.open if path.endswith(".gz") else open
    mode="rt" if path.endswith(".gz") else "r"
    with opener(path,mode,encoding="utf-8",errors="ignore") as f:
        for line in tqdm(f,desc=" metadata"):
            o=json.loads(line)
            asin=o.get("parent_asin") or o.get("asin")
            cats=o.get("categories",[])
            if asin and isinstance(cats,list) and cats:
                rows.append((asin,cats))
    df=pd.DataFrame(rows,columns=["asin","raw_categories"])
    print(f"→ {len(df):,} metadata entries")
    return df

def load_reviews(path):
    print("📥 Loading reviews…")
    rows=[]; opener=gzip.open if path.endswith(".gz") else open
    mode="rt" if path.endswith(".gz") else "r"
    with opener(path,mode,encoding="utf-8",errors="ignore") as f:
        for line in tqdm(f,desc=" reviews"):
            o=json.loads(line)
            asin=o.get("asin") or o.get("parent_asin")
            uid=o.get("user_id"); r=o.get("rating"); ts=o.get("timestamp")
            # normalize ms→s
            if isinstance(ts,(int,float)) and ts>1e12: ts=int(ts//1000)
            if asin and uid and r is not None and ts is not None:
                rows.append((asin,uid,float(r),int(ts)))
    df=pd.DataFrame(rows,columns=["asin","user_id","rating","timestamp"])
    print(f"→ {len(df):,} reviews")
    return df

def pick_genre(raw_list):
    # from each raw category find all STANDARD_GENRES containing it (or vice versa)
    cands=[]
    for rc in raw_list:
        low=rc.lower()
        for std in STANDARD_GENRES:
            if std.lower() in low or low in std.lower():
                cands.append(std)
    return random.choice(cands) if cands else "Other"

def pad_by_last(seq, L):
    if len(seq)>=L: return seq[-L:]
    return seq + [seq[-1]]*(L-len(seq))

def create_statics_file(df_rev, output_dir):
    """Create a statics.csv file similar to the first code's output"""
    num_users = df_rev['userId'].nunique()
    num_movies = df_rev['movieId'].nunique()
    num_genres = df_rev['genreId'].nunique()
    
    # Calculate average sequence lengths (all sequences are padded to RAW_SEQ_LEN-1)
    avg_movie_seq_length = RAW_SEQ_LEN - 1
    avg_genre_seq_length = RAW_SEQ_LEN - 1
    
    statics_dict = {
        "statistic": [
            "num_users", 
            "num_movies", 
            "num_genres", 
            "train_seq_length", 
            "raw_seq_length",
            "avg_movie_seq_length",
            "avg_genre_seq_length"
        ],
        "value": [
            num_users,
            num_movies,
            num_genres,
            RAW_SEQ_LEN - 1,  # train_seq_length
            RAW_SEQ_LEN,      # raw_seq_length
            avg_movie_seq_length,
            avg_genre_seq_length
        ]
    }
    
    statics_df = pd.DataFrame(statics_dict)
    statics_df.to_csv(os.path.join(output_dir, "statics.csv"), index=False)
    print("Saved statics.csv with dataset information.")

def main():
    os.makedirs(DATA_DIR,exist_ok=True)
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    mpath=os.path.join(DATA_DIR,os.path.basename(METADATA_JSONL_URL))
    rpath=os.path.join(DATA_DIR,os.path.basename(REVIEWS_JSONL_URL))
    download_if_missing(METADATA_JSONL_URL,mpath)
    download_if_missing(REVIEWS_JSONL_URL,  rpath)

    df_meta=load_metadata(mpath)
    df_rev =load_reviews(rpath)

    # 1) keep only ≥4★
    df_rev=df_rev[df_rev.rating>=RATING_THRESHOLD]
    print(f"After ≥{RATING_THRESHOLD}★: {len(df_rev):,}")

    # 2) attach raw_categories, drop missing
    mlookup=dict(zip(df_meta.asin,df_meta.raw_categories))
    df_rev["raw_categories"]=df_rev.asin.map(mlookup)
    before=len(df_rev)
    df_rev=df_rev[df_rev.raw_categories.notna()]
    print(f"After mapping cats: {len(df_rev):,}/{before:,}")

    # 3) pick & encode genre
    df_rev["genre"] = df_rev.raw_categories.map(pick_genre)
    df_rev["genreId"], genres = pd.factorize(df_rev.genre)
    print(f"→ {len(genres):,} distinct genres after mapping")

    # 4) drop low‐activity users
    uc=df_rev.user_id.value_counts()
    keep_users=uc[uc>=MIN_INTERACTIONS].index
    df_rev=df_rev[df_rev.user_id.isin(keep_users)]
    print(f"After user≥{MIN_INTERACTIONS}: {len(df_rev):,}, {df_rev.user_id.nunique():,} users")

    # 5) numeric IDs
    df_rev["userId"]  = pd.factorize(df_rev.user_id)[0]+1
    df_rev["movieId"] = pd.factorize(df_rev.asin   )[0]+1

    # Create statics.csv file
    create_statics_file(df_rev, OUTPUT_DIR)

    # 6) split users 80/10/10
    users=df_rev.userId.unique()
    np.random.shuffle(users)
    n=len(users); i80=int(0.8*n); i90=int(0.9*n)
    train_u,valid_u,test_u = users[:i80],users[i80:i90],users[i90:]
    splits=[("train",train_u),("valid",valid_u),("test",test_u)]

    # 7) build 1 sequence per user
    for name,ulist in splits:
        print(f"\n▶ Building {name} ({len(ulist)} users)…")
        out=[]
        for uid in tqdm(ulist,desc=" users"):
            sub=df_rev[df_rev.userId==uid].sort_values("timestamp")
            mids=pad_by_last(sub.movieId.tolist(),RAW_SEQ_LEN)
            gids=pad_by_last(sub.genreId.tolist(),RAW_SEQ_LEN)
            out.append({
                "movie_seq":    mids[:-1],
                "movie_len":    RAW_SEQ_LEN-1,
                "movie_target": mids[-1],
                "genre_seq":    gids[:-1],
                "genre_len":    RAW_SEQ_LEN-1,
                "genre_target": gids[-1],
            })
        df_out=pd.DataFrame(out)
        df_out.to_csv (os.path.join(OUTPUT_DIR,f"{name}.csv"), index=False)
        df_out.to_pickle(os.path.join(OUTPUT_DIR,f"{name}.df"))
        print(f"✔ Saved {name}: {len(df_out):,} sequences")

    print("\n✅ All done — outputs in",OUTPUT_DIR)

if __name__=="__main__":
    main()