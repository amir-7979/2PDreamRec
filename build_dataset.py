import argparse
import pandas as pd
from tqdm import tqdm

class MoviesDataBuilder:
    def __init__(self, rating_path='movie_dataset/rating.csv', meta_data_path='movie_dataset/movie.csv', number_of_users=None, output_path='movie_dataset/processed_dataset.csv'):
        self.rating_path = rating_path
        self.meta_data_path = meta_data_path
        self.number_of_users = number_of_users
        self.output_path = output_path
        
        self.df_rating = pd.read_csv(self.rating_path)
        self.df_meta = pd.read_csv(self.meta_data_path)
        
        self.df_rating = pd.merge(self.df_rating, self.df_meta, on='movieId')
        
        self.users = self.df_rating.userId.unique()
        self.df_rating = self.df_rating[self.df_rating.rating >= 4.0]
        
    def build_user(self):
        # Build users' preference
        self.users_pref = []
        self.movies_genres = []
        
        if self.number_of_users is not None: self.users = self.users[:self.number_of_users]
        
        for user in tqdm(self.users[:self.number_of_users], desc='Building user preferences'):
            userData = self.df_rating[self.df_rating.userId == user].sort_values('timestamp')[-11:]
            self.users_pref.append(userData.movieId.to_list())
            
            # The last item is the target, so the genres must not include its genre to prevent data leak
            genres = userData.genres.to_list()[:-1]
            genres = [",".join(item.split('|')) for item in genres]
            genres = "|".join(genres)
            self.movies_genres.append(genres)
    
    def build_dataset(self, max_length_size=10):
        features = []
        target = []
        len_seq = []
        meta_data = []
        
        for index, movies_set in tqdm(enumerate(self.users_pref), desc='Building dataset'):
            length = len(movies_set)
            if length < 2:
                continue
            
            meta_data.append(self.movies_genres[index])
            
            feature_vector = movies_set[:-1]
            target.append(movies_set[-1])
            len_seq.append((length-1))
            if max_length_size > (length-1):
                diff = max_length_size - (length-1)
                feature_vector.extend([feature_vector[-1]]*diff)
            
            assert len(feature_vector) == max_length_size, f"All of the rows must include {max_length_size} items"
            features.append(feature_vector)
            
        
        self.dataset = pd.DataFrame({
                                    'seq': features,
                                    'len_seq': len_seq,
                                    'target': target,
                                    'genres': meta_data
                                })
        
        self.dataset.to_csv(self.output_path, index=False)

class GenresDataBuilder:
    def __init__(self, processed_data_file='movie_dataset/processed_dataset.csv', movie_meta_data_path='movie_dataset/movie.csv'):
        df = pd.read_csv(processed_data_file)
        genres_output = '/'.join(processed_data_file.split('/')[:-1])+'genres.csv'
        genres_set = set()
        for row in df.to_numpy():
            genres = row[3]
            for sub_genre in genres.split('|'):
                for genre in sub_genre.split(','):
                    genres_set.add(genre)

        genres_list = list(genres_set)
        genres_id = list(zip(range(1, len(genres_list)+1), genres_list))
        self.genres_meta_data = pd.DataFrame(genres_id, columns=['genre_id', 'genre'])
        self.genres_meta_data.to_csv(genres_output, index=False)
        self.movie_meta_data = pd.read_csv(movie_meta_data_path)
        
        df = df.apply(self.encode_genres, axis=1)
        df.to_csv(processed_data_file)
        
    def encode_genres(self, row):
        genres = row['genres']
        genres_list = list()
        for sub_genre in genres.split('|'):
            genre = random.choice(sub_genre.split(','))
            genre_id = self.genres_meta_data[self.genres_meta_data.genre == genre].genre_id.values[0]
            genres_list.append(int(genre_id))
        
        while len(genres_list) < 10:
            genres_list.append(genres_list[-1])
        
        row['seq_genres'] = genres_list
        target_genres = self.movie_meta_data[self.movie_meta_data.movieId==row.target].genres.values[0].split('|')
        target_genre = random.choice(target_genres)
        target_genre_id = self.genres_meta_data[self.genres_meta_data.genre == target_genre].genre_id.values[0]
        row['target_genre'] = target_genre_id
        return row
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose what you want to generate: the initial dataset or the genres dataset.")
    parser.add_argument('--user_movie', action='store_true', default=False)
    parser.add_argument('--rating_file', default='movie_dataset/rating.csv')
    parser.add_argument('--metadata_file', default='movie_dataset/movie.csv')
    parser.add_argument('-o', '--output', default='movie_dataset/processed_dataset.csv')
    parser.add_argument('--users', default=None)
    
    parser.add_argument('--user_genres', action='store_true', default=False)
    
    
    args = parser.parse_args()
    if args.user_movie:
        assert False, args.user_movie
        data_builder = MoviesDataBuilder(rating_path=args.rating_file, meta_data_path=args.metadata_file, number_of_users=args.users, output_path=args.output)
        data_builder.build_user()
        data_builder.build_dataset()
        print(f"The initial dataset has been created!: {args.output}", flush=True)
    if args.user_genres:
        GenresDataBuilder(processed_data_file=args.output, movie_meta_data_path=args.metadata_file)
        print(f"The genres sequences were added to the dataset!: {args.output}", flush=True)