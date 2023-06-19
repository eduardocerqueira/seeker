#date: 2023-06-19T16:53:11Z
#url: https://api.github.com/gists/b1ab71354fd02a2e55ad04ef50842208
#owner: https://api.github.com/users/kitsamho

class DataLoader:
    def __init__(self, config_path):
        """
        DataLoader class for loading movie and cast data.

        Args:
            config_path (str): Path to the YAML config file. Default is 'config.yaml'.
        """
        self.config_path = config_path
        self.data_path = None
        self.cast_data = None
        self.movies_data = None
        self.cast_path = None
        self.movies_path = None
        self.df_movies = None
        self.df_cast = None
        self.df_merged = None

    def load_data(self):
        """
        Load the movie and cast data from the specified file paths in the config file.
        """
        self.read_config()
        self.construct_file_paths()
        self.df_movies = self.read_data(self.movies_path)
        self.df_cast = self.read_data(self.cast_path)
        self.df_merged = self.join_movies_cast(self.df_cast, self.df_movies)

    def read_config(self):
        """
        Read the config file and extract the data paths.
        """
        config_data = load_config(self.config_path)
        data_paths = config_data['DataPaths']
        self.data_path = data_paths['data_path']
        self.cast_data = data_paths['cast_data']
        self.movies_data = data_paths['movies_data']

    def construct_file_paths(self):
        """
        Construct the full file paths using the extracted data paths.
        """
        try:
            self.cast_path = os.path.join(self.data_path, self.cast_data)
            self.movies_path = os.path.join(self.data_path, self.movies_data)
        except Exception as e:
            raise ValueError("Error constructing file paths: {}".format(str(e)))

    def read_data(self, file_path):
        """
        Read data from the specified file path.

        Args:
            file_path (str): Path to the data file.

        Returns:
            pandas.DataFrame: Loaded data as a DataFrame.
        """
        try:
            return pd.read_pickle(file_path)
        except Exception as e:
            raise ValueError("Error reading data from {}: {}".format(file_path, str(e)))

    def join_movies_cast(self, df_cast, df_movies):
        """
        Join the cast and movies dataframes on a common column.

        Args:
            df_cast (pandas.DataFrame): Cast dataframe.
            df_movies (pandas.DataFrame): Movies dataframe.

        Returns:
            pandas.DataFrame: Merged dataframe.
        """
        try:
            return join_movies_cast(df_cast, df_movies)
        except Exception as e:
            raise ValueError("Error joining cast and movies dataframes: {}".format(str(e)))

    def get_df_merged(self):
        """
        Get the merged dataframe.

        Returns:
            pandas.DataFrame: Merged dataframe.
        """
        return self.df_merged