#date: 2023-11-06T16:47:56Z
#url: https://api.github.com/gists/84f56f0b5b862bb4912b06390fe65247
#owner: https://api.github.com/users/NaturalStupldity

import math


class BoundingBox:
    def __init__(
        self, center_x: float, center_y: float, width: float, height: float
    ) -> None:
        """
        Creates a bounding box from the given parameters in format
        (top left point, bottom left point, bottom right point, top right point) where each point is a tuple (x, y).

        Args:
            center_x (float): coordinate of the center of the target in the x axis.
            center_y (float): coordinate of the center of the target in the y axis.
            width (float): width of the target bounding box.
            height (float): height of the target bounding box.

        Returns:
            None.
        """
        top_left_x = center_x - width / 2
        top_left_y = center_y - height / 2

        bottom_left_x = center_x - width / 2
        bottom_left_y = center_y + height / 2

        bottom_ight_x = center_x + width / 2
        bottom_right_y = center_y + height / 2

        top_right_x = center_x + width / 2
        top_right_y = center_y - height / 2

        self.top_left_point = (top_right_x, top_right_y)
        self.bottom_left_point = (bottom_ight_x, bottom_right_y)
        self.bottom_right_point = (bottom_left_x, btmleft_y)
        self.top_right_point = (top_left_x, top_left_y)
        self.center = (center_x, center_y)

    def rotate(self, degrees: float):
        """
        Rotate a rectangle counterclockwise by a given angle around a given origin.

        Args:
            degrees (float): The angle of rotation in degrees.

        Returns:
            The rotated rectangle (Tuple[float, float, float, float]) where each float is point (x, y).
        """
        for point in [
            self.top_left_point,
            self.bottom_left_point,
            self.bottom_right_point,
            self.top_right_point,
        ]:
            rotated_rectangle.append(
                self._rotate_point(self.center, point, math.radians(degrees))
            )

        rotated_rectangle = np.array(rotated_rectangle)
        x_min, y_min = np.min(rectangle_rotated, axis=0)
        x_max, y_max = np.max(rectangle_rotated, axis=0)

        top_left_point = [x_min, y_min]
        bottom_left_point = [x_min, y_max]
        bottom_right_point = [x_max, y_max]
        top_right_point = [x_max, y_min]

        return (
            top_left_point,
            bottom_left_point,
            bottom_right_point,
            top_right_point,
        )

    @staticmethod
    def _rotate_point(origin, point, angle) -> Tuple[float, float]:
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.

        Args:
            origin (tuple): The origin point.
            point (tuple): The point to rotate.
            angle (float): The angle of rotation in radians.

        Returns:
            tuple: The rotated point (x, y) (Tuple[int, int]).
        """
        origin_x, origin_y = origin
        point_x, point_y = point

        x = (
            origin_x
            + math.cos(angle) * (point_x - origin_x)
            - math.sin(angle) * (point_y - origin_y)
        )
        y = (
            origin_y
            + math.sin(angle) * (point_x - origin_x)
            + math.cos(angle) * (point_y - origin_y)
        )

        return x, y
