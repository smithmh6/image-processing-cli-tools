"""
The rectangle.py module contains the classes needed
for creating a rectangle from mouse event coordinates
on an image opened with OpenCV.
"""

# import dependencies
from math import sqrt

class CoordinateError(Exception):
    """
    An exception class to handle invalid
    coordinates in the Point() class.
    """

    def __init__(self, *args):
        """
        Handle a custom input message, if necessary.
        """
        if args:
            try:
                self.message = args[0]
            except TypeError:
                raise TypeError('Argument must be of type str().')
        else:
            self.message = None

    def __str__(self):
        """
        Define how the exception is displayed.
        """
        if self.message:
            return 'CoordinateError, {0}'.format(self.message)
        else:
            return 'CoordinateError, (x, y) must of of type int().'

class Point():
    """
    An object that represents a pixel location
    as an (x, y) coordinate on an image.
    """
    x = None
    y = None

    def __init__(self, x:int, y:int):
        """
        Initialize a Point object.
        """
        try:
            self.x = int(x)
            self.y = int(y)
        except:
            raise CoordinateError("Could not cast values to integers.")

    def __str__(self) -> str:
        return f'({self.x}, {self.y})'

    def __repr__(self) -> str:
        return f'Point({self.x}, {self.y})'

    def __add__(self, object):
        return Point(self.x + object.x, self.y + object.y)

    def __radd__(self, object):
        return Point(object.x + self.x, object.y + self.y)

    def __sub__(self, object):
        return Point(self.x - object.x, self.y - object.y)

    def __rsub__(self, object):
        return Point(object.x - self.x, object.y - self.y)

    def __mul__(self, object):
        return self.x * object.x + self.y * object.y

    def __rmul__(self, object):
        return object.x * self.x + object.y * self.y

    def euclid_distance(self, p_in:'Point') -> float:
        """
        Calculates the Euclidean distance between <Point> object
        and an input <Point>.

        Return
        -----------
        (float) Euclidean distance between the two points.

        Example
        -----------
        >>> p1 = Point(x1, y1)
        >>> p2 = Point(x2, y2)
        >>> dist1 = p1.euclid_distance(p2)
        >>> dist2 = p2.euclid_distance(p1)
        """

        return sqrt((p_in.x - self.x)**2 + (p_in.y - self.y)**2)



class Rectangle():
    """
    Abstract representation of a rectangle. Requires 2 Point()
    arguments to fully define the rectangle.
    """

    # start and end coordinates
    point1 = None
    point2 = None

    # dimensions
    width = None
    height = None

    # vertex coordinates
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None
    set_corners = False

    def __init__(self, p1:Point, p2:Point, set_corners=True):
        """
        Initialize the rectangle by calculating the
        width and height and setting the vertices.
        """

        self.point1 = p1
        self.point2 = p2

        self.width = abs(p2.x - p1.x)
        self.height = abs(p2.y - p1.y)

        if set_corners:
            self.set_vertices()

    def __repr__(self) -> str:
        return f'Rectangle(Point({self.point1.x}, {self.point1.y}), Point({self.point2.x}, {self.point2.y}))'

    def __str__(self) -> str:
        return (
            f'[({self.top_left.x}, {self.top_left.y}),'
            + f' ({self.top_right.x}, {self.top_right.y}),'
            + f' ({self.bottom_left.x}, {self.bottom_left.y}),'
            + f' ({self.bottom_right.x}, {self.bottom_right.y})]'
        )

    def __iter__(self):
        """
        Override for iterable functions like list().
        """
        return iter([
            [self.top_left.x, self.top_left.y],
            [self.top_right.x, self.top_right.y],
            [self.bottom_left.x, self.bottom_left.y],
            [self.bottom_right.x, self.bottom_right.y]])

    def get_area(self) -> int:
        """
        Return Height x Width
        """
        return int(self.height * self.width)

    def get_size(self) -> tuple:
        """
        Return a tuple of form (Width, Height)
        """
        return (self.width, self.height)

    def get_vertices(self) -> list:
        """
        Return a list of vertices (Point(x, y) tuples) in
        the order [top_left, top_right, bottom_left, bottom_right].
        """
        return [
            self.top_left, self.top_right,
            self.bottom_left, self.bottom_right
        ]

    def set_vertices(self):
        """
        Set the vertices of the rectangle.
        """

        # scenario 1
        if self.point1.x < self.point2.x and self.point1.y < self.point2.y:
            self.top_left = Point(self.point1.x, self.point1.y)
            self.top_right = Point(self.point2.x, self.point1.y)
            self.bottom_left = Point(self.point1.x, self.point2.y)
            self.bottom_right = Point(self.point2.x, self.point2.y)

        # scenario 2
        elif self.point1.x < self.point2.x and self.point1.y > self.point2.y:
            self.top_left = Point(self.point1.x, self.point2.y)
            self.top_right = Point(self.point2.x, self.point2.y)
            self.bottom_left = Point(self.point1.x, self.point1.y)
            self.bottom_right = Point(self.point2.x, self.point1.y)

        # scenario 3
        elif self.point1.x > self.point2.x and self.point1.y < self.point2.y:
            self.top_left = Point(self.point2.x, self.point1.y)
            self.top_right = Point(self.point1.x, self.point1.y)
            self.bottom_left = Point(self.point2.x, self.point2.y)
            self.bottom_right = Point(self.point1.x, self.point2.y)

        # scenario 4
        elif self.point1.x > self.point2.x and self.point1.y > self.point2.y:
            self.top_left = Point(self.point2.x, self.point2.y)
            self.top_right = Point(self.point1.x, self.point2.y)
            self.bottom_left = Point(self.point2.x, self.point1.y)
            self.bottom_right = Point(self.point1.x, self.point1.y)

        self.set_corners = True

