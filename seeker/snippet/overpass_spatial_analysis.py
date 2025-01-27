#date: 2025-01-27T16:54:06Z
#url: https://api.github.com/gists/ec116701c65fcedc35f1081535af4786
#owner: https://api.github.com/users/aniqu18

# Uses Overpass API to retrieve the data from OSM

import overpass
import pandas as pd
from shapely.geometry import Polygon


class SpatialAnalysis:
    def __init__(self):
        self.api = overpass.API(timeout=30)

    @staticmethod
    def _show_value_in_df(words: list[str], df: pd.DataFrame, column_index: int = 3) -> pd.DataFrame:
        """
        Filters the DataFrame based on the presence of specific words in a given column.

        Args:
            words (list[str]): A list of words to search for in the DataFrame column.
            df (pd.DataFrame): The DataFrame to filter.
            column_index (int, optional): The index of the column to search in. Defaults to 3.

        Returns:
            pd.DataFrame: The filtered DataFrame containing only the rows where the words are found.
        """
        if not isinstance(words, list):
            raise ValueError('Argument must be a list')
        bool_values = [False] * len(df.iloc[:, column_index])
        for word in words:
            for i, value in enumerate(df.iloc[:, column_index]):
                if str(value).__contains__(word):
                    bool_values[i] = True
        return df[bool_values]

    def _compute_area_for_factor(self, space_df: pd.DataFrame, factors: list[str]) -> dict[str, int]:
        """
        Compute the factor area based on the given space dataframe, factor, and AQ dictionary.

        Parameters:
        - space_df (pd.DataFrame): The dataframe containing spatial data.
        - factor (str): The factor to compute the area for.

        Returns:
        - dict[str,int]: The computed factors' areas.
        """

        map_areas = {}

        for factor in factors:

            factor_area = 0  # will be returned by the function

            values = self._show_value_in_df(words=[factor], df=space_df)

            if values['properties'].any():
                for i in values.itertuples(index=False):
                    type_ = i[0]
                    if type_ == 'Polygon':
                        factor_area += self._compute_area_from_coords(i[2]['coordinates'])
                    elif type_ == 'Line':
                        pass
                    elif type_ == 'Node':
                        pass
                    else:
                        raise Exception('Type must be of Node, Line or Polygon')

            # rounding
            factor_area = round(factor_area)

            map_areas[factor] = factor_area

        return map_areas

    @staticmethod
    def _compute_area_from_coords(geometry: list[list[float]]) -> float:
        """
        Estimates area based on polygon coordinates.

        Parameters:
        - geometry (list[list[float]]): A list of coordinates representing the polygon.

        Returns:
        - float: The estimated area.
        """
        polygon = Polygon(geometry)
        polygon_area = polygon.area
        polygon_area = polygon_area * (10 ** 7)
        return polygon_area

    @staticmethod
    def _get_cords_overpass(cords: list) -> str:
        """
        Returns the coordinates in the correct order for the Overpass API.

        Parameters:
        - cords (list): A list of coordinates.

        Returns:
        - str: The coordinates in the correct order.
        """
        cords_ordered = [cords[1], cords[0], cords[3], cords[2]]
        cords_str = ','.join([str(i) for i in cords_ordered])
        return cords_str

    def _get_node_df(self, cords: str) -> pd.DataFrame:
        """
        Get the nodes dataframe based on the given coordinates.

        Parameters:
        - cords (str): The coordinates to get the nodes for.

        Returns:
        - pd.DataFrame: The nodes dataframe.
        """
        pass_request = f'node({cords});'
        response = self.api.get(pass_request)
        df = pd.DataFrame(response['features'])
        df['type'] = 'Node'
        return df

    def _get_way_df(self, cords: str) -> pd.DataFrame:
        """
        Get the way dataframe based on the given coordinates.

        Parameters:
        - cords (str): The coordinates to get the ways for.

        Returns:
        - pd.DataFrame: The ways dataframe.
        """
        pass_request = f'way({cords});'
        pass_request = pass_request + ' out geom;'
        response = self.api.get(pass_request)
        df = pd.DataFrame(response['features'])
        df['type'] = 'Way'
        return df

    @staticmethod
    def _check_geometry(row) -> str:
        """
        Check the geometry of the row and return the type. It can be 'Delete', 'Polygon', 'Line', or 'Node', where
        'Delete' means that the geometry is empty.

        Parameters:
        - row (pd.Series): The row to check the geometry for.

        Returns:
        - str: The type of the geometry.
        """
        type_ = row['type']
        geometry = row['geometry']
        if geometry == {'type': 'LineString', 'coordinates': []}:
            return 'Delete'
        elif type_ == 'Way':
            coords = geometry['coordinates']
            if coords[0] == coords[-1]:
                return 'Polygon'
            else:
                return 'Line'
        else:
            return 'Node'

    def _get_space_df(
            self,
            west_limit: float,
            south_limit: float,
            east_limit: float,
            north_limit: float
    ) -> pd.DataFrame:
        """
        Get the space dataframe based on the given limits. Space dataframe contains nodes and ways from the Overpass
        API response.

        Parameters:
        - west_limit (float): The western limit of the area.
        - south_limit (float): The southern limit of the area.
        - east_limit (float): The eastern limit of the area.
        - north_limit (float): The northern limit of the area.

        Returns:
        - pd.DataFrame: The space dataframe.
        """
        cords = self._get_cords_overpass([west_limit, south_limit, east_limit, north_limit])
        df_nodes = self._get_node_df(cords)
        df_ways = self._get_way_df(cords)
        df_combined = pd.concat([df_nodes, df_ways], axis=0)
        space_df = df_combined[df_combined['properties'] != {}]
        space_df.loc[:, 'type'] = space_df.apply(self._check_geometry, axis=1)
        space_df = space_df[space_df['type'] != 'Delete']
        space_df.reset_index(drop=True, inplace=True)
        return space_df

    def get_area(
            self,
            west_limit: float,
            south_limit: float,
            east_limit: float,
            north_limit: float,
            factors: list[str]
    ) -> dict[str, int]:
        """
        Calculate the area based on the given limits and factor.

        Parameters:
        west_limit (float): The western limit of the area.
        south_limit (float): The southern limit of the area.
        east_limit (float): The eastern limit of the area.
        north_limit (float): The northern limit of the area.
        factor (str): The factor used to compute the area (for example 'forest')

        Returns:
        dict[str, int]: The calculated area for the given factor.

        """
        space_df = self._get_space_df(west_limit, south_limit, east_limit, north_limit)
        calculated_area = self._compute_area_for_factor(space_df=space_df, factors=factors)
        return calculated_area

    @staticmethod
    def get_proper_bounding_box(y_coord: float, x_coord: float) -> tuple[float, float, float, float]:
        """
        Adds/subtracts 0.02 to/from the given coordinates to get a proper bounding box.

        Parameters:
        - y_coord (float): The y-coordinate.
        - x_coord (float): The x-coordinate.

        Returns:
        - tuple[float, float, float, float]: The bounding box.
        """
        west_limit = x_coord - 0.02
        east_limit = x_coord + 0.02
        south_limit = y_coord - 0.02
        north_limit = y_coord + 0.02
        return west_limit, east_limit, south_limit, north_limit

    def _compute_score_for_factor(self, dict_key: str, factors_dict: dict[str, int], space_df: pd.DataFrame) -> float:
        """
        Function calculates impact of given factors on air quality by computing the area of the factor and multiplying
        it by the factor's weight.

        Parameters:
        - dict_key (str): The factor name.
        - factors_dict (dict[str, int]): The dictionary containing factors and their weights.
        - space_df (pd.DataFrame): The dataframe containing spatial data.

        Returns:
        - float: The computed score.
        """

        area = 0
        values = self._show_value_in_df(words=[dict_key], df=space_df)

        if values['properties'].any():
            for i in values.itertuples(index=False):
                type_ = i[0]
                if type_ == 'Node':
                    pass
                elif type_ == 'Line':
                    pass
                elif type_ == 'Polygon':
                    area += self._compute_area_from_coords(i[2]['coordinates'])
                else:
                    raise Exception('Type must be of Node, Line or Polygon')

        score = 0
        score += area * factors_dict[dict_key]

        return score

    def _compute_spatial_score(self, space_df: pd.DataFrame, factors: dict) -> float:
        """
        Function calculates impact of given factors on air quality.

        Parameters:
        - space_df (pd.DataFrame): The dataframe containing spatial data.

        Returns:
        - float: The computed score.
        """

        score = 0

        for factor in factors:
            score += self._compute_score_for_factor(dict_key=factor, factors_dict=factors, space_df=space_df)

        # rounding
        score = round(score)
        return score

    def get_spatial_score(
            self,
            west_limit: float,
            south_limit: float,
            east_limit: float,
            north_limit: float,
            dict_factors: dict[str, int],
    ) -> float:
        """
        Makes assessment of a given space impact on the air quality.

        Parameters:
        - west_limit (float): The western limit of the area.
        - south_limit (float): The southern limit of the area.
        - east_limit (float): The eastern limit of the area.
        - north_limit (float): The northern limit of the area.
        - dict_factors (dict[str, int]): The dictionary containing factors and their weights.

        Returns:
        - float: The computed score.
        """
        space_df = self._get_space_df(west_limit, south_limit, east_limit, north_limit)
        score = self._compute_spatial_score(space_df, dict_factors)

        return score
