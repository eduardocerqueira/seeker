#date: 2023-06-19T16:56:58Z
#url: https://api.github.com/gists/95ea9cdc2bec0ef829d85ee499162b82
#owner: https://api.github.com/users/kitsamho

class MovieCastTransformer:
    """
    A class for processing movie data based on user-selected filters.

    Args:
        merged_df (pandas.DataFrame): The merged dataframe containing movie and cast data.

    Attributes:
        merged_df (pandas.DataFrame): The merged dataframe containing movie and cast data.
        year_start (int): The start year selected by the user.
        year_end (int): The end year selected by the user.
        gender_choice (str): The gender choice selected by the user.

    """

    def __init__(self, merged_df):
        self.merged_df = merged_df
        self.year_start, self.year_end = self.__get_year_range()
        self.gender_choice = self.__select_gender()

    def __get_year_range(self):
        """
        Get the range of movie years selected by the user.

        Returns:
            Tuple[int, int]: The start and end year selected by the user.

        """
        st.sidebar.subheader('Movie Filters')
        min_year, max_year = get_min_max_values(self.merged_df, 'm_release_year', int)
        year_start, year_end = st.sidebar.slider('Year of release', min_year, max_year, (2018, 2023), 1)
        return year_start, year_end

    def __select_gender(self):
        """
        Select the gender choice for filtering the cast data.

        Returns:
            str: The gender choice selected by the user.

        """
        return select_gender(st.sidebar.selectbox('Gender', ['Everyone', 'Male', 'Female']))

    def __filter_data(self):
        """
        Filter the movie and cast data based on user-selected filters.

        """
        # Filter movie data based on year and average rating
        self.merged_df = mask_range(self.merged_df, 'm_release_year', self.year_start, self.year_end)
        self.merged_df = select_movie_data(self.merged_df, self.year_start, self.year_end)

        # Filter cast data based on actor popularity and gender
        self.merged_df = select_cast_data(self.merged_df, self.gender_choice)

        # Perform any additional data processing or transformations

    def transform_data(self):
        """
        Process the movie data by filtering and transforming it based on user-selected filters.

        Returns:
            pandas.DataFrame: The processed dataframe containing the filtered movie and cast data.

        """
        # Filter and process data
        self.__filter_data()

        # Return processed data
        return self.merged_df