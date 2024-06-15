"""Various vector utilities"""

import itertools
import math
from dataclasses import dataclass
from typing import (
    Any,
    FrozenSet,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import glm
import numpy as np
import skimage.segmentation
from glm import bvec2, bvec3, ivec2, ivec3, vec2, vec3
from more_itertools import powerset
from scipy import ndimage
from typing_extensions import Protocol

from .utils import nonZeroSign

# ==================================================================================================
# VecLike Protocols
# ==================================================================================================


class Vec2iLike(Protocol):
    """Protocol for a vector that contains two integers."""

    def __getitem__(self, __i: int) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...


class Vec3iLike(Protocol):
    """Protocol for a vector that contains three integers."""

    def __getitem__(self, __i: int) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...


class Vec2bLike(Protocol):
    """Protocol for a vector that contains two bools."""

    def __getitem__(self, __i: int) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[bool]: ...


class Vec3bLike(Protocol):
    """Protocol for a vector that contains three bools."""

    def __getitem__(self, __i: int) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[bool]: ...


# ==================================================================================================
# Constants
# ==================================================================================================


# ==== 2D values ====


# == constants ==


ZERO_2D = ivec2(0, 0)
X_2D = ivec2(1, 0)
Y_2D = ivec2(0, 1)
XY_2D: ivec2 = X_2D + Y_2D

EAST_2D:  ivec2 = X_2D
WEST_2D:  ivec2 = -EAST_2D
SOUTH_2D: ivec2 = Y_2D
NORTH_2D: ivec2 = -SOUTH_2D

NORTHWEST_2D: ivec2 = NORTH_2D + WEST_2D
NORTHEAST_2D: ivec2 = NORTH_2D + EAST_2D
SOUTHEAST_2D: ivec2 = SOUTH_2D + EAST_2D
SOUTHWEST_2D: ivec2 = SOUTH_2D + WEST_2D

CARDINALS_2D:               FrozenSet[ivec2] = frozenset({NORTH_2D, SOUTH_2D, EAST_2D, WEST_2D})
INTERCARDINALS_2D:          FrozenSet[ivec2] = frozenset({NORTHEAST_2D, NORTHWEST_2D, SOUTHEAST_2D, SOUTHWEST_2D})
CARDINALS_AND_DIAGONALS_2D: FrozenSet[ivec2] = CARDINALS_2D | INTERCARDINALS_2D
DIAGONALS_2D              = tuple(INTERCARDINALS_2D)  # NOTE: Legacy format

# starting East, moving clockwise
# NOTE: Use `utils.rotateSequence(...)` to start at a different point
ORDERED_CARDINALS_2D:               Tuple[ivec2, ...] = (EAST_2D, SOUTH_2D, WEST_2D, NORTH_2D)
ORDERED_INTERCARDINALS_2D:          Tuple[ivec2, ...] = (SOUTHEAST_2D, SOUTHWEST_2D, NORTHWEST_2D, NORTHEAST_2D)
ORDERED_CARDINALS_AND_DIAGONALS_2D: Tuple[ivec2, ...] = tuple(itertools.chain.from_iterable(zip(ORDERED_CARDINALS_2D, ORDERED_INTERCARDINALS_2D)))


# ==== 3D values ====


# == functions for generating constants ==


def _spiraloidDirections3D(
        top_pattern:     Optional[Tuple[ivec3, ...]],
        center_pattern:  Optional[Tuple[ivec3, ...]],
        bottom_pattern:  Optional[Tuple[ivec3, ...]],
        include_up:      bool = False,
        include_center:  bool = False,
        include_down:    bool = False
    ) -> Generator[ivec3, None, None]:
    """yields 3D direction vectors of a spiraloid, where patterns can be provided to be combined with a top, center and bottom vector."""

    # If desired, yields...
    if include_up:     yield UP_3D                                        # ...the UP vector...
    if top_pattern:    yield from (UP_3D + c for c in top_pattern)        # ...the upward diagonal vectors...
    if center_pattern: yield from center_pattern[:len(center_pattern)//2] # ...the first half of the horizontal vectors...
    if include_center: yield ZERO_3D                                      # ...the origin...
    if center_pattern: yield from center_pattern[len(center_pattern)//2:] # ...the second half of the horizontal vectors...
    if bottom_pattern: yield from (DOWN_3D + c for c in bottom_pattern)   # ...the downward diagonal vectors...
    if include_down:   yield DOWN_3D                                      # ...and the DOWN vector.


def _symmetricSpiraloidDirections3D(
        top_and_bottom_pattern: Optional[Tuple[ivec3, ...]],
        central_pattern:        Optional[Tuple[ivec3, ...]],
        include_up_and_down:    bool = False,
        include_center:         bool = False,
    ) -> Generator[ivec3, None, None]:
    """Yields 3D direction vectors of a spiraloid, mirrored across the XZ-plane."""
    yield from _spiraloidDirections3D(
        top_pattern     = top_and_bottom_pattern,
        center_pattern  = central_pattern,
        bottom_pattern  = top_and_bottom_pattern,
        include_up      = include_up_and_down,
        include_center  = include_center,
        include_down    = include_up_and_down
    )


# == constants ==


ZERO_3D = ivec3(0, 0, 0)

X_3D = ivec3(1, 0, 0)
Y_3D = ivec3(0, 1, 0)
Z_3D = ivec3(0, 0, 1)

XY_3D: ivec3 = X_3D + Y_3D
XZ_3D: ivec3 = X_3D + Z_3D
YZ_3D: ivec3 = Y_3D + Z_3D

XYZ_3D: ivec3 = X_3D + Y_3D + Z_3D

UP_3D:    ivec3 = Y_3D
DOWN_3D:  ivec3 = -UP_3D
EAST_3D:  ivec3 = X_3D
WEST_3D:  ivec3 = -EAST_3D
SOUTH_3D: ivec3 = Z_3D
NORTH_3D: ivec3 = -SOUTH_3D

NORTHEAST_3D: ivec3 = NORTH_3D + EAST_3D
NORTHWEST_3D: ivec3 = NORTH_3D + WEST_3D
SOUTHWEST_3D: ivec3 = SOUTH_3D + WEST_3D
SOUTHEAST_3D: ivec3 = SOUTH_3D + EAST_3D

CARDINALS_3D:               FrozenSet[ivec3] = frozenset({NORTH_3D, SOUTH_3D, EAST_3D, WEST_3D})
INTERCARDINALS_3D:          FrozenSet[ivec3] = frozenset({NORTHEAST_3D, NORTHWEST_3D, SOUTHEAST_3D, SOUTHWEST_3D})
CARDINALS_AND_DIAGONALS_3D: FrozenSet[ivec3] = CARDINALS_3D | INTERCARDINALS_3D

# starting East, moving clockwise
# NOTE: Use `utils.rotateSequence(ORDERED_..., n)` to change the starting point while maintaining the order of the sequence
#       E.g. n=1 starts at the second point; n=-1 starts at the last point
# NOTE: Use `reversed(utils.rotateSequence(ORDERED_...))` to move counterclockwise from east
#       This does not work for sequences with differing Y-values!
#       To achieve that, transform the values for each layer first, then recombine them.
#       E.g. `reverse(utils.rotateSequence([UP_3D + c for c in ORDERED_CARDINALS_3D])) + reverse(utils.rotateSequence(ORDERED_CARDINALS_3D)) + ...`
ORDERED_CARDINALS_3D:               Tuple[ivec3, ...] = (EAST_3D, SOUTH_3D, WEST_3D, NORTH_3D)
ORDERED_INTERCARDINALS_3D:          Tuple[ivec3, ...] = (SOUTHEAST_3D, SOUTHWEST_3D, NORTHWEST_3D, NORTHEAST_3D)
ORDERED_CARDINALS_AND_DIAGONALS_3D: Tuple[ivec3, ...] = tuple(itertools.chain.from_iterable(zip(ORDERED_CARDINALS_3D, ORDERED_INTERCARDINALS_3D)))

DIRECTIONS_3D:     FrozenSet[ivec3] = CARDINALS_3D | {UP_3D, DOWN_3D}
EDGE_DIAGONALS_3D: FrozenSet[ivec3] = INTERCARDINALS_3D | {
    verticality + cardinal
    for verticality, cardinal in itertools.product((UP_3D, DOWN_3D), CARDINALS_3D)
}
DIRECTIONS_AND_EDGE_DIAGONALS_3D: FrozenSet[ivec3] = DIRECTIONS_3D | EDGE_DIAGONALS_3D
CORNER_DIAGONALS_3D:              FrozenSet[ivec3] = frozenset({
    verticality + cardinal
    for verticality, cardinal in itertools.product((UP_3D, DOWN_3D), INTERCARDINALS_3D)
})
DIRECTIONS_AND_ALL_DIAGONALS_3D: FrozenSet[ivec3] = DIRECTIONS_AND_EDGE_DIAGONALS_3D | CORNER_DIAGONALS_3D
# TODO: tuple() for backwards compatibility. Remove on major release
DIAGONALS_3D                   = tuple(EDGE_DIAGONALS_3D | CORNER_DIAGONALS_3D)  # NOTE: Legacy format

# Moving Up to Down, clockwise starting East
# NOTE: For other combinations, use `generate_[symmetric_]spiraloid_vectors_3D()`
ORDERED_DIRECTIONS_3D:                    Tuple[ivec3, ...] = (UP_3D, *ORDERED_CARDINALS_3D, DOWN_3D)
ORDERED_EDGE_DIAGONALS_3D:                Tuple[ivec3, ...] = tuple(_symmetricSpiraloidDirections3D(ORDERED_CARDINALS_3D,               ORDERED_INTERCARDINALS_3D                                   ))
ORDERED_DIRECTIONS_AND_EDGE_DIAGONALS_3D: Tuple[ivec3, ...] = tuple(_symmetricSpiraloidDirections3D(ORDERED_CARDINALS_3D,               ORDERED_CARDINALS_AND_DIAGONALS_3D, include_up_and_down=True))
ORDERED_CORNER_DIAGONALS_3D:              Tuple[ivec3, ...] = tuple(_symmetricSpiraloidDirections3D(ORDERED_INTERCARDINALS_3D,          None                                                        ))
ORDERED_DIRECTIONS_AND_ALL_DIAGONALS_3D:  Tuple[ivec3, ...] = tuple(_symmetricSpiraloidDirections3D(ORDERED_CARDINALS_AND_DIAGONALS_3D, ORDERED_CARDINALS_AND_DIAGONALS_3D, include_up_and_down=True))
ORDERED_DIAGONALS:                        Tuple[ivec3, ...] = tuple(_symmetricSpiraloidDirections3D(ORDERED_CARDINALS_AND_DIAGONALS_3D, ORDERED_INTERCARDINALS_3D                                   ))


# ==== aliases ====


ZERO: ivec3 = ZERO_3D
X: ivec3 = X_3D
Y: ivec3 = Y_3D
Z: ivec3 = Z_3D
XY: ivec3 = XY_3D
XZ: ivec3 = XZ_3D
YZ: ivec3 = YZ_3D
XYZ: ivec3 = XYZ_3D
UP :   ivec3 = UP_3D
DOWN:  ivec3 = DOWN_3D
EAST:  ivec3 = EAST_3D
WEST:  ivec3 = WEST_3D
SOUTH: ivec3 = SOUTH_3D
NORTH: ivec3 = NORTH_3D
NORTHEAST: ivec3 = NORTHEAST_3D
NORTHWEST: ivec3 = NORTHWEST_3D
SOUTHWEST: ivec3 = SOUTHWEST_3D
SOUTHEAST: ivec3 = SOUTHEAST_3D
CARDINALS:               FrozenSet[ivec3] = CARDINALS_3D
INTERCARDINALS:          FrozenSet[ivec3] = INTERCARDINALS_3D
CARDINALS_AND_DIAGONALS: FrozenSet[ivec3] = CARDINALS_AND_DIAGONALS_3D
EDGE_DIAGONALS:                FrozenSet[ivec3] = EDGE_DIAGONALS_3D
CORNER_DIAGONALS:              FrozenSet[ivec3] = CORNER_DIAGONALS_3D
DIRECTIONS:                    FrozenSet[ivec3] = DIRECTIONS_3D
DIRECTIONS_AND_EDGE_DIAGONALS: FrozenSet[ivec3] = DIRECTIONS_AND_EDGE_DIAGONALS_3D
DIRECTIONS_AND_ALL_DIAGONALS:  FrozenSet[ivec3] = DIRECTIONS_AND_ALL_DIAGONALS_3D
ORDERED_CARDINALS:               Tuple[ivec3, ...] = ORDERED_CARDINALS_3D
ORDERED_INTERCARDINALS:          Tuple[ivec3, ...] = ORDERED_INTERCARDINALS_3D
ORDERED_CARDINALS_AND_DIAGONALS: Tuple[ivec3, ...] = ORDERED_CARDINALS_AND_DIAGONALS_3D
ORDERED_EDGE_DIAGONALS:                Tuple[ivec3, ...] = ORDERED_EDGE_DIAGONALS_3D
ORDERED_CORNER_DIAGONALS:              Tuple[ivec3, ...] = ORDERED_CORNER_DIAGONALS_3D
ORDERED_DIRECTIONS:                    Tuple[ivec3, ...] = ORDERED_DIRECTIONS_3D
ORDERED_DIRECTIONS_AND_EDGE_DIAGONALS: Tuple[ivec3, ...] = ORDERED_DIRECTIONS_AND_EDGE_DIAGONALS_3D
ORDERED_DIRECTIONS_AND_ALL_DIAGONALS:  Tuple[ivec3, ...] = ORDERED_DIRECTIONS_AND_ALL_DIAGONALS_3D
DIAGONALS: tuple = DIAGONALS_3D


# ==================================================================================================
# General
# ==================================================================================================


def dropDimension(vec: Vec3iLike, dimension: int) -> ivec2:
    """Returns <vec> without its <dimension>-th component"""
    if dimension == 0: return ivec2(vec[1], vec[2])
    if dimension == 1: return ivec2(vec[0], vec[2])
    if dimension == 2: return ivec2(vec[0], vec[1])
    raise ValueError(f'Invalid dimension "{dimension}"')


def addDimension(vec: Vec2iLike, dimension: int, value: int = 0) -> ivec3:
    """Inserts <value> into <vec> at <dimension> and returns the resulting 3D vector"""
    # NOTE: Should be adjusted to only support 2D -> 3D, or all ivec dimensions
    l = list(vec)
    return ivec3(*l[:dimension], value, *l[dimension:])


def dropY(vec: Vec3iLike) -> ivec2:
    """Returns [vec] without its y-component (i.e., projected on the XZ-plane)"""
    return ivec2(vec[0], vec[2])


def addY(vec: Vec2iLike, y=0) -> ivec3:
    """Returns a 3D vector (vec[0], y, vec[1])"""
    return ivec3(vec[0], y, vec[1])


def setY(vec: Vec3iLike, y=0) -> ivec3:
    """Returns [vec] with its y-component set to [y]"""
    return ivec3(vec[0], y, vec[2])


def trueMod2D(vec: Vec2iLike, modulus: int) -> ivec2:
    """Returns <v> modulo <modulus>.\n
    Negative numbers are handled just like Python's built-in integer modulo."""
    return ivec2(vec[0] % modulus, vec[1] % modulus)


def trueMod3D(vec: Vec3iLike, modulus: int) -> ivec3:
    """Returns <v> modulo <modulus>.\n
    Negative numbers are handled just like Python's built-in integer modulo."""
    return ivec3(vec[0] % modulus, vec[1] % modulus, vec[2] % modulus)


def perpendicular(vec: Vec2iLike) -> ivec2:
    """Returns the vector perpendicular to [vec] that points to the right of [vec] and has the same
    length as [vec]."""
    return ivec2(vec[1], -vec[0])


def rotate2D(vec: Vec2iLike, rotation: int) -> ivec2:
    """Returns [vec], rotated clockwise by [rotation] quarters."""
    if rotation == 0: return ivec2(*vec)
    if rotation == 1: return ivec2(-vec[1],  vec[0])
    if rotation == 2: return ivec2(-vec[0], -vec[1])
    if rotation == 3: return ivec2( vec[1], -vec[0])
    raise ValueError("Rotation must be in {0,1,2,3}")


def rotate3D(vec: Vec3iLike, rotation: int) -> ivec3:
    """Returns [vec], rotated clockwise in the XZ-plane by [rotation] quarters."""
    return addY(rotate2D(dropY(vec), rotation), vec[1])


def rotate2Ddeg(vec: Vec2iLike, degrees: int) -> ivec2:
    """
    Rotate a 2D vector clockwise by a specified number of degrees.

    Args:
        vec: The 2D vector to rotate.
        degrees: The number of degrees to rotate the vector by. Only ±90°-rotations and their multiples are valid.

    Returns:
        The rotated 2D vector.

    Raises:
        ValueError: If degrees is not a multiple of 90.

    Examples:
        vec = ivec2(0, 1)
        rotated_vec = rotate2Ddeg(vec, -90)
    """

    if degrees % 90 != 0:
        raise ValueError("Only ±90°-rotations and their multiples are valid!")

    rotation: int = (degrees // 90) % 4  # Convert to quarter-turns

    return rotate2D(vec, rotation)


def rotate3Ddeg(vec: Vec3iLike, degrees: int) -> ivec3:
    """
    Rotate a 3D vector clockwise by a specified number of degrees through the Y axis.

    Args:
        vec: The 3D vector to rotate.
        degrees: The number of degrees to rotate the vector by through the Y axis. Only ±90°-rotations and their multiples are valid.

    Returns:
        The rotated 3D vector.

    Raises:
        ValueError: If degrees is not a multiple of 90.

    Examples:
        vec = ivec3(0, 1)
        rotated_vec = rotate3Ddeg(vec, -90)
    """
    return addY(rotate2Ddeg(dropY(vec), degrees), vec[1])

def flipRotation2D(rotation: int, flip: Vec2bLike) -> int:
    """Returns rotation such that applying rotation after <flip> is equivalent to applying <flip>
    after <rotation>."""
    scale: ivec2 = flipToScale2D(flip)
    return (rotation * scale.x * scale.y + 4) % 4


def flipRotation3D(rotation: int, flip: Vec3bLike) -> int:
    """Returns rotation such that applying rotation after <flip> is equivalent to applying <flip>
    after <rotation>"""
    return flipRotation2D(rotation, dropY(flip))


def rotateSize2D(size: Vec2iLike, rotation: int) -> Vec2iLike:
    """Returns the effective size of a rect of size [size] that has been rotated in the XZ-plane by
    [rotation]."""
    return ivec2(size[1], size[0]) if rotation in {1, 3} else size


def rotateSize3D(size: Vec3iLike, rotation: int) -> ivec3:
    """Returns the effective size of a box of size [size] that has been rotated in the XZ-plane by
    [rotation]."""
    return addY(rotateSize2D(dropY(size), rotation), size[1])


def flipToScale2D(flip: Vec2bLike) -> ivec2:
    """Returns a vector with a 1 where <flip> is false, and -1 where <flip> is true"""
    return 1 - 2 * ivec2(*flip)


def flipToScale3D(flip: Vec3bLike) -> ivec3:
    """Returns a vector with a 1 where <flip> is false, and -1 where <flip> is true"""
    return 1 - 2 * ivec3(*flip)


def scaleToFlip2D(scale: Vec2iLike) -> bvec2:
    """Returns whether [scale] flips space in each axis"""
    return bvec2(scale[0] < 0, scale[1] < 0)


def scaleToFlip3D(scale: Vec3iLike) -> bvec3:
    """Returns whether [scale] flips space in each axis"""
    return bvec3(scale[0] < 0, scale[1] < 0, scale[2] < 0)


def toAxisVector2D(vec: Vec2iLike) -> ivec2:
    """Returns the axis-aligned unit vector closest to [vec]"""
    if abs(vec[0]) > abs(vec[1]):  # pylint: disable=no-else-return
        return ivec2(nonZeroSign(vec[0]), 0)
    else:
        return ivec2(0, nonZeroSign(vec[1]))


def directionToRotation(direction: Vec2iLike) -> Union[Literal[0], Literal[1], Literal[2], Literal[3]]:
    """Returns the rotation that rotates (0,-1) closest to [direction]"""
    vec = toAxisVector2D(direction)
    if vec[1] < 0: return 0
    if vec[0] > 0: return 1
    if vec[1] > 0: return 2
    if vec[0] < 0: return 3
    raise ValueError()


# For some reason, glm's length, length2, distance, distance2 and l1Norm refuse to work with integer
# vectors. We provide some wrappers.


def length(vec: Union[Vec2iLike, Vec3iLike]) -> float:
    """Returns the length of [vec]"""
    if len(vec) == 2: return glm.length(vec2(*vec))
    if len(vec) == 3: return glm.length(vec3(*vec))
    raise ValueError()


def length2(vec: Union[Vec2iLike, Vec3iLike]) -> int:
    """Returns the squared length of [vec]"""
    if len(vec) == 2: return int(glm.length2(vec2(*vec)))
    if len(vec) == 3: return int(glm.length2(vec3(*vec)))
    raise ValueError()


def distance(vecA: Union[Vec2iLike, Vec3iLike], vecB: Union[Vec2iLike, Vec3iLike]) -> float:
    """Returns the distance between [vecA] and [vecB]"""
    if len(vecA) == 2 and len(vecB) == 2: return glm.distance(vec2(*vecA), vec2(*vecB))
    if len(vecA) == 3 and len(vecB) == 3: return glm.distance(vec3(*vecA), vec3(*vecB))
    raise ValueError()


def distance2(vecA: Union[Vec2iLike, Vec3iLike], vecB: Union[Vec2iLike, Vec3iLike]) -> int:
    """Returns the squared distance between [vecA] and [vecB]"""
    if len(vecA) == 2 and len(vecB) == 2: return int(glm.distance2(vec2(*vecA), vec2(*vecB)))
    if len(vecA) == 3 and len(vecB) == 3: return int(glm.distance2(vec3(*vecA), vec3(*vecB)))
    raise ValueError()


def l1Norm(vec: Union[Vec2iLike, Vec3iLike]) -> int:
    """Returns the L1 norm of [vec]"""
    return sum(abs(n) for n in vec)


def l1Distance(vecA: Union[Vec2iLike, Vec3iLike], vecB: Union[Vec2iLike, Vec3iLike]) -> int:
    """Returns the L1 norm distance between [vecA] and [vecB]"""
    return l1Norm(vecA - vecB)


# End of glm wrappers.


def orderedCorners2D(corner1: Vec2iLike, corner2: Vec2iLike) -> Tuple[ivec2, ivec2]:
    """Returns two corners of the rectangle defined by <corner1> and <corner2>, such that the first
    corner is smaller than the second corner in each axis"""
    return (
        ivec2(
            corner1[0] if corner1[0] <= corner2[0] else corner2[0],
            corner1[1] if corner1[1] <= corner2[1] else corner2[1],
        ),
        ivec2(
            corner1[0] if corner1[0] > corner2[0] else corner2[0],
            corner1[1] if corner1[1] > corner2[1] else corner2[1],
        ),
    )


def orderedCorners3D(corner1: Vec3iLike, corner2: Vec3iLike) -> Tuple[ivec3, ivec3]:
    """Returns two corners of the box defined by <corner1> and <corner2>, such that the first
    corner is smaller than the second corner in each axis"""
    return (
        ivec3(
            corner1[0] if corner1[0] <= corner2[0] else corner2[0],
            corner1[1] if corner1[1] <= corner2[1] else corner2[1],
            corner1[2] if corner1[2] <= corner2[2] else corner2[2],
        ),
        ivec3(
            corner1[0] if corner1[0] > corner2[0] else corner2[0],
            corner1[1] if corner1[1] > corner2[1] else corner2[1],
            corner1[2] if corner1[2] > corner2[2] else corner2[2],
        ),
    )


def getDimensionality(corner1: Union[Vec2iLike, Vec3iLike], corner2: Union[Vec2iLike, Vec3iLike]) -> Tuple[int, List[str]]:
    """Determines the number of dimensions for which <corner1> and <corner2> are in general
    position, i.e. the number of dimensions for which the volume they define is not flat.\n
    Returns (dimensionality, list of indices of dimensions for which the volume is flat).
    For example: (2, [0,2]) means that the volume is flat in the x and z axes."""
    difference = np.array(corner1) - np.array(corner2)
    flatSides: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.argwhere(difference == 0).flatten()
    return int(len(corner1) - np.sum(flatSides)), list(flatSides)


# ==================================================================================================
# Rect and Box
# ==================================================================================================


# TODO: If someone knows how to fix the duplication in Rect and Box, please do tell.


@dataclass
class Rect:
    """A rectangle, defined by an offset and a size"""

    _offset: ivec2
    _size: ivec2

    def __init__(self, offset: Vec2iLike = (0, 0), size: Vec2iLike = (0, 0)) -> None:
        self._offset = ivec2(*offset)
        self._size = ivec2(*size)

    def __hash__(self) -> int:
        return hash((self.offset, self.size))

    def __repr__(self) -> str:
        return f"Rect({tuple(self._offset)}, {tuple(self._size)})"

    def __iter__(self) -> ivec2:
        dx, dy = self.size.x, self.size.y
        for x, y in itertools.product(range(dx), range(dy)):
            yield self.begin + ivec2(x, y)

    @property
    def offset(self) -> ivec2:
        """This Rect's offset"""
        return self._offset

    @offset.setter
    def offset(self, value: Vec2iLike) -> None:
        self._offset = ivec2(*value)

    @property
    def size(self) -> ivec2:
        """This Rect's size"""
        return self._size

    @size.setter
    def size(self, value: Vec2iLike) -> None:
        self._size = ivec2(*value)

    @property
    def begin(self) -> ivec2:
        """Equivalent to self.offset. Setting will modify self.offset."""
        return self._offset

    @begin.setter
    def begin(self, value: Vec2iLike) -> None:
        self._offset = ivec2(*value)

    @property
    def end(self) -> ivec2:
        """Equivalent to self.offset + self.size. Setting will modify self.size."""
        return self.begin + self._size

    @end.setter
    def end(self, value: Vec2iLike) -> None:
        self._size = ivec2(*value) - self.begin

    @property
    def last(self) -> ivec2:
        """Equivalent to self.offset + self.size - 1. Setting will modify self.size."""
        return self._offset + self._size - 1

    @last.setter
    def last(self, value: Vec2iLike) -> None:
        self._size = ivec2(*value) - self._offset + 1

    @property
    def middle(self) -> ivec2:
        """This Rect's middle point, rounded down"""
        return self._offset + self._size // 2

    @property
    def center(self) -> ivec2:
        """Equivalent to .middle"""
        return self.middle

    @property
    def inner(self) -> Generator[ivec2, None, None]:
        """Yields all points contained in this Rect"""
        return (
            ivec2(x, y)
            for x in range(self.begin.x, self.end.x)
            for y in range(self.begin.y, self.end.y)
        )

    @property
    def area(self) -> int:
        """This Rect's surface area"""
        return self._size.x * self._size.y

    @property
    def corners(self) -> Generator[ivec2, None, None]:
        """Yields this Rect's corner points"""
        return (
            self._offset + sum(subset)
            for subset in powerset(
                [ivec2(self._size.x - 1, 0), ivec2(0, self._size.y - 1)]
            )
        )

    def contains(self, vec: Vec2iLike) -> bool:
        """Returns whether this Rect contains [vec]"""
        return (
            self.begin.x <= vec[0] < self.end.x and self.begin.y <= vec[1] < self.end.y
        )

    def collides(self, other: "Rect") -> bool:
        """Returns whether this Rect and [other] have any overlap"""
        return (
                self.begin.x <= other.end  .x
            and self.end  .x >= other.begin.x
            and self.begin.y <= other.end  .y
            and self.end  .y >= other.begin.y
        )

    def squaredDistanceToVec(self, vec: Vec2iLike) -> int:
        """Returns the squared distance between this Rect and [vec]"""
        dx: int = max(self.begin.x - vec[0], 0, vec[0] - (self.end[0] - 1))
        dy: int = max(self.begin.y - vec[1], 0, vec[1] - (self.end[1] - 1))
        return dx**2 + dy**2

    def distanceToVec(self, vec: Vec2iLike) -> float:
        """Returns the distance between this Rect and [vec]"""
        return math.sqrt(self.squaredDistanceToVec(vec))

    def translated(self, translation: Union[Vec2iLike, int]) -> "Rect":
        """Returns a copy of this Rect, translated by [translation]"""
        return Rect(self._offset + ivec2(*translation), self._size)

    def dilate(self, dilation: int = 1) -> None:
        """Morphologically dilates this rect by [dilation]"""
        self._offset -= dilation
        self._size += dilation * 2

    def dilated(self, dilation: int = 1) -> "Rect":
        """Returns a copy of this Rect, morphologically dilated by [dilation]"""
        return Rect(self._offset - dilation, self._size + dilation * 2)

    def erode(self, erosion: int = 1) -> None:
        """Morphologically erodes this rect by [erosion]"""
        self.dilate(-erosion)

    def eroded(self, erosion: int = 1) -> "Rect":
        """Returns a copy of this Rect, morphologically eroded by [erosion]"""
        return self.dilated(-erosion)

    def centeredSubRectOffset(self, size: Vec2iLike) -> ivec2:
        """Returns an offset such that Rect(offset, [size]).middle == self.middle"""
        difference: ivec2 = self._size - ivec2(*size)
        return self._offset + difference / 2

    def centeredSubRect(self, size: Vec2iLike) -> "Rect":
        """Returns a rect of size [size] with the same middle as this rect"""
        return Rect(self.centeredSubRectOffset(size), size)

    @staticmethod
    def between(cornerA: Vec2iLike, cornerB: Vec2iLike) -> "Rect":
        """Returns the Rect between [cornerA] and [cornerB] (inclusive),
        which may be any opposing corners."""
        first, last = orderedCorners2D(cornerA, cornerB)
        return Rect(first, (last - first) + 1)

    @staticmethod
    def bounding(points: Iterable[Vec2iLike]) -> "Rect":
        """Returns the smallest Rect containing all [points]"""
        pointArray = np.fromiter(points, dtype=np.dtype((int, 2)))
        minPoint = np.min(pointArray, axis=0)
        maxPoint = np.max(pointArray, axis=0)
        return Rect(minPoint, maxPoint - minPoint + 1)

    def toBox(self, offsetY=0, sizeY=0) -> "Box":
        """Returns a corresponding Box"""
        return Box(addY(self.offset, offsetY), addY(self._size, sizeY))

    @property
    def outline(self) -> Generator[ivec2, Any, None]:
        """Yields this Rect's outline points"""
        # It's surprisingly difficult to get this right without duplicates. (Think of the corners!)
        first = self.begin
        last = self.end - 1
        yield from loop2D(ivec2(first.x, first.y), ivec2(last.x  - 1, first.y    ) + 1)
        yield from loop2D(ivec2(last.x,  first.y), ivec2(last.x,      last.y  - 1) + 1)
        yield from loop2D(ivec2(last.x,  last.y ), ivec2(first.x + 1, last.y     ) - 1)
        yield from loop2D(ivec2(first.x, last.y ), ivec2(first.x,     first.y + 1) - 1)


@dataclass()
class Box:
    """A box, defined by an offset and a size"""

    _offset: ivec3
    _size: ivec3

    def __init__(self, offset: Vec3iLike = (0, 0, 0), size: Vec3iLike = (0, 0, 0)) -> None:
        self._offset = ivec3(*offset)
        self._size = ivec3(*size)

    def __hash__(self) -> int:
        return hash((self.offset, self.size))

    def __repr__(self) -> str:
        return f"Box({tuple(self._offset)}, {tuple(self._size)})"

    def __iter__(self) -> ivec3:
        dx, dy, dz = self.size.x, self.size.y, self.size.z
        for x, y in itertools.product(range(dx), range(dy), range(dz)):
            yield self.begin + ivec3(x, y)

    @property
    def offset(self) -> ivec3:
        """This Box's offset"""
        return self._offset

    @offset.setter
    def offset(self, value: Vec3iLike) -> None:
        self._offset = ivec3(*value)

    @property
    def size(self) -> ivec3:
        """This Box's size"""
        return self._size

    @size.setter
    def size(self, value: Vec3iLike) -> None:
        self._size = ivec3(*value)

    @property
    def begin(self) -> ivec3:
        """Equivalent to self.offset. Setting will modify self.offset."""
        return self._offset

    @begin.setter
    def begin(self, value: Vec3iLike) -> None:
        self._offset = ivec3(*value)

    @property
    def end(self) -> ivec3:
        """Equivalent to self.offset + self.size. Setting will modify self.size."""
        return self.begin + self._size

    @end.setter
    def end(self, value: Vec3iLike) -> None:
        self._size = ivec3(*value) - self.begin

    @property
    def last(self) -> ivec3:
        """Equivalent to self.offset + self.size - 1. Setting will modify self.size."""
        return self._offset + self._size - 1

    @last.setter
    def last(self, value: Vec3iLike) -> None:
        self._size = ivec3(*value) - self._offset + 1

    @property
    def middle(self) -> ivec3:
        """This Box's middle point, rounded down"""
        return self.begin + self._size // 2

    @property
    def center(self) -> ivec3:
        """Equivalent to .middle"""
        return self.middle

    @property
    def inner(self) -> Generator[ivec3, None, None]:
        """Yields all points contained in this Box"""
        return (
            ivec3(x, y, z)
            for x in range(self.begin.x, self.end.x)
            for y in range(self.begin.y, self.end.y)
            for z in range(self.begin.z, self.end.z)
        )

    @property
    def volume(self) -> int:
        """This Box's volume"""
        return self._size.x * self._size.y * self._size.z

    @property
    def corners(self) -> List[ivec3]:
        """Yields this Box's corner points"""
        return [
            self._offset + sum(subset)
            for subset in powerset(
                [
                    ivec3(self._size.x - 1, 0, 0),
                    ivec3(0, self._size.y - 1, 0),
                    ivec3(0, 0, self._size.z - 1),
                ]
            )
        ]

    def contains(self, vec: Vec3iLike) -> bool:
        """Returns whether this Box contains [vec]"""
        return (
                self.begin.x <= vec[0] < self.end.x
            and self.begin.y <= vec[1] < self.end.y
            and self.begin.z <= vec[2] < self.end.z
        )

    def collides(self, other: "Box") -> bool:
        """Returns whether this Box and [other] have any overlap"""
        return (
                self.begin.x <= other.end  .x
            and self.end  .x >= other.begin.x
            and self.begin.y <= other.end  .y
            and self.end  .y >= other.begin.y
            and self.begin.z <= other.end  .z
            and self.end  .z >= other.begin.z
        )

    def squaredDistanceToVec(self, vec: Vec3iLike) -> int:
        """Returns the squared distance between this Box and [vec]"""
        dx: int = max(self.begin.x - vec[0], 0, vec[0] - (self.end.x - 1))
        dy: int = max(self.begin.y - vec[1], 0, vec[1] - (self.end.y - 1))
        dz: int = max(self.begin.z - vec[2], 0, vec[2] - (self.end.z - 1))
        return dx**2 + dy**2 + dz**2

    def distanceToVec(self, vec: Vec3iLike) -> float:
        """Returns the distance between this Box and [vec]"""
        return math.sqrt(self.squaredDistanceToVec(vec))

    def translated(self, translation: Union[Vec3iLike, int]) -> "Box":
        """Returns a copy of this Box, translated by [translation]"""
        return Box(self._offset + ivec3(*translation), self._size)

    def dilate(self, dilation: int = 1) -> None:
        """Morphologically dilates this box by [dilation]"""
        self._offset -= dilation
        self._size += dilation * 2

    def dilated(self, dilation: int = 1) -> "Box":
        """Returns a copy of this Box, morphologically dilated by [dilation]"""
        return Box(self._offset - dilation, self._size + dilation * 2)

    def erode(self, erosion: int = 1) -> None:
        """Morphologically erodes this box by [erosion]"""
        self.dilate(-erosion)

    def eroded(self, erosion: int = 1) -> "Box":
        """Returns a copy of this Box, morphologically eroded by [erosion]"""
        return self.dilated(-erosion)

    def centeredSubBoxOffset(self, size: Vec3iLike) -> ivec3:
        """Returns an offset such that Box(offset, [size]).middle == self.middle"""
        difference: ivec3 = self._size - ivec3(*size)
        return self._offset + difference / 2

    def centeredSubBox(self, size: Vec3iLike) -> "Box":
        """Returns an box of size [size] with the same middle as this box"""
        return Box(self.centeredSubBoxOffset(size), size)

    @staticmethod
    def between(cornerA: Vec3iLike, cornerB: Vec3iLike) -> "Box":
        """Returns the Box between [cornerA] and [cornerB] (both inclusive),
        which may be any opposing corners"""
        first, last = orderedCorners3D(cornerA, cornerB)
        return Box(first, last - first + 1)

    @staticmethod
    def bounding(points: Iterable[Vec3iLike]) -> "Box":
        """Returns the smallest Box containing all [points]"""
        pointArray = np.fromiter(points, dtype=np.dtype((int, 3)))
        minPoint: np.ndarray = np.min(pointArray, axis=0)
        maxPoint: np.ndarray = np.max(pointArray, axis=0)
        return Box(minPoint, maxPoint - minPoint + 1)

    def toRect(self) -> Rect:
        """Returns this Box's XZ-plane as a Rect"""
        return Rect(dropY(self._offset), dropY(self._size))

    @property
    def shell(self) -> Generator[ivec3, Any, None]:
        """Yields all points on this Box's surface"""
        # It's surprisingly difficult to get this right without duplicates. (Think of the corners!)
        first: ivec3 = self.begin
        last: ivec3 = self.end - 1
        # Bottom face
        yield from loop3D(
            ivec3(first.x, first.y, first.z),
            ivec3(last.x, first.y, last.z) + 1,
        )
        # Top face
        yield from loop3D(
            ivec3(first.x, last.y, first.z),
            ivec3(last.x, last.y, last.z) + 1,
        )
        # Sides
        if self._size.y < 3:
            return
        yield from loop3D(
            ivec3(first.x, first.y + 1, first.z),
            ivec3(last.x - 1, last.y - 1, first.z) + 1,
        )
        yield from loop3D(
            ivec3(last.x, first.y + 1, first.z),
            ivec3(last.x, last.y - 1, last.z - 1) + 1,
        )
        yield from loop3D(
            ivec3(last.x, first.y + 1, last.z),
            ivec3(first.x + 1, last.y + 1, last.z) - 1,
        )
        yield from loop3D(
            ivec3(first.x, first.y + 1, last.z),
            ivec3(first.x, last.y + 1, first.z + 1) - 1,
        )

    @property
    def wireframe(self) -> Generator[ivec3, Any, None]:
        """Yields all points on this Box's edges"""
        # It's surprisingly difficult to get this right without duplicates. (Think of the corners!)
        first: ivec3 = self.begin
        last: ivec3 = self.end - 1
        # Bottom face
        yield from loop3D(
            ivec3(first.x, first.y, first.z),
            ivec3(last.x - 1, first.y, first.z) + 1,
        )
        yield from loop3D(
            ivec3(last.x, first.y, first.z),
            ivec3(last.x, first.y, last.z - 1) + 1,
        )
        yield from loop3D(
            ivec3(last.x, first.y, last.z),
            ivec3(first.x + 1, first.y, last.z) - 1,
        )
        yield from loop3D(
            ivec3(first.x, first.y, last.z),
            ivec3(first.x, first.y, first.z + 1) - 1,
        )
        # top face
        yield from loop3D(
            ivec3(first.x, last.y, first.z),
            ivec3(last.x - 1, last.y, first.z) + 1,
        )
        yield from loop3D(
            ivec3(last.x, last.y, first.z),
            ivec3(last.x, last.y, last.z - 1) + 1,
        )
        yield from loop3D(
            ivec3(last.x, last.y, last.z),
            ivec3(first.x + 1, last.y, last.z) - 1,
        )
        yield from loop3D(
            ivec3(first.x, last.y, last.z),
            ivec3(first.x, last.y, first.z + 1) - 1,
        )
        # sides
        if self._size.y < 3:
            return
        yield from loop3D(
            ivec3(first.x, first.y + 1, first.z),
            ivec3(first.x, last.y - 1, first.z) + 1,
        )
        yield from loop3D(
            ivec3(last.x, first.y + 1, first.z),
            ivec3(last.x, last.y - 1, first.z) + 1,
        )
        yield from loop3D(
            ivec3(last.x, first.y + 1, last.z),
            ivec3(last.x, last.y - 1, last.z) + 1,
        )
        yield from loop3D(
            ivec3(first.x, first.y + 1, last.z),
            ivec3(first.x, last.y - 1, last.z) + 1,
        )


def rectSlice(array: np.ndarray, rect: Rect) -> np.ndarray:
    """Returns the slice from [array] defined by [rect]"""
    return array[rect.begin.x : rect.end.x, rect.begin.y : rect.end.y]


def setRectSlice(array: np.ndarray, rect: Rect, value: Any) -> None:
    """Sets the slice from [array] defined by [rect] to [value]"""
    array[rect.begin.x : rect.end.x, rect.begin.y : rect.end.y] = value


def boxSlice(array: np.ndarray, box: Box) -> np.ndarray:
    """Returns the slice from [array] defined by [box]"""
    return array[box.begin.x : box.end.x, box.begin.y : box.end.y, box.begin.z : box.end.z]


def setBoxSlice(array: np.ndarray, box: Box, value: Any) -> None:
    """Sets the slice from [array] defined by [box] to [value]"""
    array[box.begin.x : box.end.x, box.begin.y : box.end.y, box.begin.z : box.end.z] = value


# ==================================================================================================
# Point generation
# ==================================================================================================


def loop2D(begin: Vec2iLike, end: Optional[Vec2iLike] = None) -> Generator[ivec2, Any, None]:
    """Yields all points between <begin> and <end> (end-exclusive).\n
    If <end> is not given, yields all points between (0,0) and <begin>."""
    if end is None: begin, end = (0, 0), begin

    for x in range(begin[0], end[0], nonZeroSign(end[0] - begin[0])):
        for y in range(begin[1], end[1], nonZeroSign(end[1] - begin[1])):
            yield ivec2(x, y)


def loop3D(begin: Vec3iLike, end: Optional[Vec3iLike] = None) -> Generator[ivec3, Any, None]:
    """Yields all points between <begin> and <end> (end-exclusive).\n
    If <end> is not given, yields all points between (0,0,0) and <begin>."""
    if end is None:
        begin, end = (0, 0, 0), begin

    for x in range(begin[0], end[0], nonZeroSign(end[0] - begin[0])):
        for y in range(begin[1], end[1], nonZeroSign(end[1] - begin[1])):
            for z in range(begin[2], end[2], nonZeroSign(end[2] - begin[2])):
                yield ivec3(x, y, z)


def cuboid2D(corner1: Vec2iLike, corner2: Vec2iLike) -> Generator[ivec2, None, None]:
    """Yields all points in the rectangle between <corner1> and <corner2> (inclusive)."""
    return Rect.between(corner1, corner2).inner


def cuboid3D(corner1: Vec3iLike, corner2: Vec3iLike) -> Generator[ivec3, None, None]:
    """Yields all points in the box between <corner1> and <corner2> (inclusive)."""
    return Box.between(corner1, corner2).inner


def filled2DArray(
    points: Iterable[Vec2iLike],
    seedPoint: Vec2iLike,
    boundingRect: Optional[Rect] = None,
    includeInputPoints=True,
) -> np.ndarray:
    """Fills the shape defined by <points>, starting at <seedPoint> and returns a (n,2) numpy array
    containing the resulting points.\n
    <boundingRect> should contain all <points>. If not provided, it is calculated."""
    if boundingRect is None:
        boundingRect = Rect.bounding(points)

    pointMap = np.zeros(boundingRect.size.to_tuple(), dtype=int)
    pointMap[
        tuple(np.transpose(np.fromiter(points, dtype=np.dtype((int, 2))) - np.array(boundingRect.offset)))] = 1
    filled = skimage.segmentation.flood_fill(pointMap, tuple(ivec2(*seedPoint) - boundingRect.offset),1, footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if not includeInputPoints:
        filled -= pointMap
    return np.argwhere(filled) + np.array(boundingRect.offset)


def filled2D(
    points: Iterable[Vec2iLike],
    seedPoint: Vec2iLike,
    boundingRect: Optional[Rect] = None,
    includeInputPoints=True,
) -> Generator[ivec2, None, None]:
    """Fills the shape defined by <points>, starting at <seedPoint> and yields the resulting points.\n
    <boundingRect> should contain all <points>. If not provided, it is calculated."""
    return (
        ivec2(*point)
        for point in filled2DArray(points, seedPoint, boundingRect, includeInputPoints)
    )


def filled3DArray(
    points: Iterable[Vec3iLike],
    seedPoint: Vec3iLike,
    boundingBox: Optional[Box] = None,
    includeInputPoints=True,
) -> np.ndarray:
    """Fills the shape defined by <points>, starting at <seedPoint> and returns a (n,3) numpy array
    containing the resulting points.\n
    <boundingBox> should contain all <points>. If not provided, it is calculated."""
    if boundingBox is None:
        boundingBox = Box.bounding(points)

    pointMap = np.zeros(boundingBox.size.to_tuple(), dtype=int)
    pointMap[tuple(np.transpose(np.fromiter(points, dtype=np.dtype((int, 3))) - np.array(boundingBox.offset)))] = 1
    filled = skimage.segmentation.flood_fill(pointMap, tuple(ivec3(*seedPoint) - boundingBox.offset), 1, connectivity=1)
    if not includeInputPoints:
        filled -= pointMap
    return np.argwhere(filled) + np.array(boundingBox.offset)


def filled3D(
    points: Iterable[Vec3iLike],
    seedPoint: Vec3iLike,
    boundingBox: Optional[Box] = None,
    includeInputPoints=True,
) -> Generator[ivec3, None, None]:
    """Fills the shape defined by <points>, starting at <seedPoint> and yields the resulting points.\n
    <boundingBox> should contain all <points>. If not provided, it is calculated."""
    return (
        ivec3(*point)
        for point in filled3DArray(points, seedPoint, boundingBox, includeInputPoints)
    )


# TODO: separate out thickening code?
def _lineArray(begin: Union[Vec2iLike, Vec3iLike], end: Union[Vec2iLike, Vec3iLike], width: int = 1) -> np.ndarray:
    begin = np.array(begin)
    end = np.array(end)
    delta = end - begin
    maxDelta = int(max(abs(delta)))
    if maxDelta == 0:
        return np.array([])
    points = delta[np.newaxis, :] * np.arange(maxDelta + 1)[:, np.newaxis] / maxDelta + np.array(begin)
    points = np.rint(points).astype(np.signedinteger)

    if width > 1:
        minPoint = np.minimum(begin, end)

        # convert point array to a map
        array_width: int = maxDelta + width * 2
        array = np.zeros([array_width] * len(begin), dtype=int)
        array[tuple(np.transpose(points - minPoint + width))] = 1

        array = ndimage.binary_dilation(array, iterations=width - 1)

        # rebuild point array from map
        points = np.argwhere(array) + minPoint - width

    return points


def line2DArray(begin: Vec2iLike, end: Vec2iLike, width: int = 1) -> np.ndarray:
    """Returns (n,2) numpy array of points on the line between [begin] and [end] (inclusive)"""
    return _lineArray(begin, end, width)


def line2D(begin: Vec2iLike, end: Vec2iLike, width: int = 1) -> Generator[ivec2, None, None]:
    """Yields the points on the line between [begin] and [end] (inclusive)"""
    return (ivec2(*point) for point in _lineArray(begin, end, width))


def line3Darray(begin: Vec3iLike, end: Vec3iLike, width: int = 1) -> np.ndarray:
    """Returns (n,3) numpy array of points on the line between [begin] and [end] (inclusive)"""
    return _lineArray(begin, end, width)


def line3D(begin: Vec3iLike, end: Vec3iLike, width: int = 1) -> Generator[ivec3, None, None]:
    """Yields the points on the line between [begin] and [end] (inclusive)"""
    return (ivec3(*point) for point in _lineArray(begin, end, width))


def lineSequence2D(points: Iterable[Vec2iLike], closed=False) -> Generator[ivec2, Any, None]:
    """Yields all points on the lines that connect <points>"""
    pointList = list(points)
    for i in range((-1 if closed else 0), len(pointList) - 1):
        yield from line2D(pointList[i], pointList[i + 1])


def lineSequence3D(points: Iterable[Vec3iLike], closed=False) -> Generator[ivec3, Any, None]:
    """Yields all points on the lines that connect <points>"""
    pointList = list(points)
    for i in range((-1 if closed else 0), len(pointList) - 1):
        yield from line3D(pointList[i], pointList[i + 1])


def circle(center: Vec2iLike, diameter: int, filled=False):
    """Yields the points of the specified circle.\n
    If <diameter> is even, <center> will be the bottom left center point."""

    # With 'inspiration' from:
    # https://www.geeksforgeeks.org/bresenhams-circle-drawing-algorithm/

    center: ivec2 = ivec2(*center)

    if diameter == 0:
        empty: List[ivec2] = []
        return iter(empty)

    e: int = 1 - (diameter % 2)  # for even centers
    points: Set[ivec2] = set()

    def eightPoints(x: int, y: int):
        points.add(center + ivec2(e + x, e + y))
        points.add(center + ivec2(0 - x, e + y))
        points.add(center + ivec2(e + x, 0 - y))
        points.add(center + ivec2(0 - x, 0 - y))
        points.add(center + ivec2(e + y, e + x))
        points.add(center + ivec2(0 - y, e + x))
        points.add(center + ivec2(e + y, 0 - x))
        points.add(center + ivec2(0 - y, 0 - x))

    radius: int = (diameter - 1) // 2
    x, y = 0, radius
    d: int = 3 - 2 * radius
    eightPoints(x, y)
    while y >= x:
        # for each pixel we will
        # draw all eight pixels

        x += 1

        # check for decision parameter
        # and correspondingly
        # update d, x, y
        if d > 0:
            y -= 1
            d = d + 4 * (x - y) + 10
        else:
            d = d + 4 * x + 6
        eightPoints(x, y)

    if filled:
        return filled2D(
            points, center, Rect(center - radius, ivec2(diameter, diameter))
        )
    return iter(points)


def fittingCircle(corner1: Vec2iLike, corner2: Vec2iLike, filled=False):
    """Yields the points of the largest circle that fits between <corner1> and <corner2>.\n
    The circle will be centered in the larger axis."""
    corner1_, corner2_ = orderedCorners2D(corner1, corner2)
    diameter: int = min(corner2_ - corner1_) + 1
    return circle((corner1_ + corner2_) // 2, diameter, filled)


def ellipse(center: Vec2iLike, diameters: Vec2iLike, filled=False):
    """Yields the points of the specified ellipse.\n
    If <diameter>[axis] is even, <center>[axis] will be the lower center point in that axis."""

    # Modified version 'inspired' by chandan_jnu from
    # https://www.geeksforgeeks.org/midpoint-ellipse-drawing-algorithm/

    center: ivec2 = ivec2(*center)
    diameters: ivec2 = ivec2(*diameters)

    if diameters.x == 0 or diameters.y == 0:
        empty: List[ivec2] = []
        return iter(empty)

    if diameters.x == diameters.y:
        return circle(center, diameters.x, filled)

    e: ivec2 = 1 - (diameters % 2)

    points: Set[ivec2] = set()

    def fourpoints(x, y) -> None:
        points.add(center + ivec2(e.x + x, e.y + y))
        points.add(center + ivec2(    - x, e.y + y))
        points.add(center + ivec2(e.x + x,     - y))
        points.add(center + ivec2(    - x,     - y))

        if filled:
            points.update(line2D(center + ivec2(-x, e.y + y), center + ivec2(e.x + x, e.y + y)))
            points.update(line2D(center + ivec2(-x,     - y), center + ivec2(e.x + x,     - y)))

    rx, ry = (diameters - 1) // 2

    x, y = 0, ry

    # Initial decision parameter of region 1
    d1: float = (ry**2) - (rx**2 * ry) + (0.25 * rx**2)
    dx: int = 2 * ry**2 * x
    dy: int = 2 * rx**2 * y

    # For region 1
    while dx < dy:
        fourpoints(x, y)

        # Checking and updating value of
        # decision parameter based on algorithm
        if d1 < 0:
            x += 1
            dx += (2 * ry**2)
            d1 += dx + (ry**2)
        else:
            x += 1
            y -= 1
            dx += (2 * ry**2)
            dy -= (2 * rx**2)
            d1 = d1 + dx - dy + (ry**2)

    # Decision parameter of region 2
    d2: float = (
        ((ry**2) * ((x + 0.5) * (x + 0.5)))
        + ((rx**2) * ((y - 1) * (y - 1)))
        - (rx**2 * ry**2)
    )

    # Plotting points of region 2
    while y >= 0:
        fourpoints(x, y)

        # Checking and updating parameter
        # value based on algorithm
        if d2 > 0:
            y -= 1
            dy = dy - (2 * rx * rx)
            d2 = d2 + (rx * rx) - dy
        else:
            y -= 1
            x += 1
            dx = dx + (2 * ry * ry)
            dy = dy - (2 * rx * rx)
            d2 = d2 + dx - dy + (rx * rx)

    return iter(points)


def fittingEllipse(corner1: Vec2iLike, corner2: Vec2iLike, filled=False):
    """Yields the points of the largest ellipse that fits between <corner1> and <corner2>."""
    _corner1, _corner2 = orderedCorners2D(corner1, corner2)
    diameters: ivec2 = (_corner2 - _corner1) + 1
    return ellipse((_corner1 + _corner2) // 2, diameters, filled)


def cylinder(
    baseCenter: Vec3iLike,
    diameters: Union[Vec2iLike, int],
    length: int,
    axis=1,
    tube=False,
    hollow=False,
):
    """Yields the points from the specified cylinder.\n
    If a <diameter> is even, <center> will be the lower center point in that axis.\n
    <tube> has precedence over <hollow>."""

    diameters: ivec2 = (
        ivec2(diameters) if isinstance(diameters, int) else ivec2(*diameters)
    )
    baseCenter: ivec3 = ivec3(*baseCenter)

    if diameters.x == 0 or diameters.y == 0 or length == 0:
        empty: List[ivec3] = []
        return iter(empty)

    corner1 = baseCenter - addDimension((diameters - 1) / 2, axis, 0)
    corner2 = corner1 + addDimension(diameters - 1, axis, length - 1)
    return fittingCylinder(corner1, corner2, axis, tube, hollow)


def fittingCylinder(
    corner1: Vec3iLike, corner2: Vec3iLike, axis=1, tube=False, hollow=False
):
    """Yields the points of the largest cylinder that fits between <corner1> and <corner2>.\n
    <tube> has precedence over <hollow>."""

    _corner1, _corner2 = orderedCorners3D(corner1, corner2)
    dimensionality, flatSides = getDimensionality(_corner1, _corner2)

    if dimensionality == 0:
        yield _corner1
        return

    if dimensionality == 1 or (dimensionality == 2 and flatSides[0] != axis):
        yield from cuboid3D(_corner1, _corner2)
        return

    baseCorner1: ivec2 = dropDimension(_corner1, axis)
    baseCorner2: ivec2 = dropDimension(_corner2, axis)
    h0: int = _corner1[axis]
    hn: int = _corner2[axis]

    ellipsePoints2D = list(fittingEllipse(baseCorner1, baseCorner2, filled=False))
    ellipsePoints3D: List[ivec3] = [addDimension(point, axis, h0) for point in ellipsePoints2D]

    basePoints: List[ivec3] = ellipsePoints3D
    bodyPoints: List[ivec3] = ellipsePoints3D

    if not tube:
        basePoints = [
            addDimension(point, axis, h0)
            for point in filled2D(
                ellipsePoints2D,
                (baseCorner1 + baseCorner2) // 2,
                Rect.between(baseCorner1, baseCorner2),
            )
        ]
        bodyPoints = ellipsePoints3D if hollow else basePoints

    yield from basePoints
    if hn != h0:
        direction = ivec3(0, 0, 0)
        direction[axis] = 1
        yield from (point + (hn - h0) * direction for point in basePoints)
        yield from (
            point + i * direction
            for i in range(1, hn - h0)
            for point in bodyPoints
        )


def ellipsoid(center: Vec3iLike, diameters: Vec3iLike, hollow: bool = False) -> Generator[ivec3, Any, None]:
    """Yields the points of an ellipsoid centered on <center> with diameters <diameters>.\n
    If <diameter>[axis] is even, <center>[axis] will be the lower center point in that axis."""

    # Convert the center and diameters to ivec3
    center: ivec3 = ivec3(*center)
    diameters: ivec3 = ivec3(*diameters)

    # Calculate the correction
    e: ivec3 = 1 - (diameters % 2)

    def are_points_in_line(center: Vec3iLike, point: Vec3iLike) -> bool:
        """Checks if two 3D points are the same on 2 or more axis"""
        count = 0
        for i in range(3):
            if point[i] == center[i]: count += 1
        return count >= 2

    def generate_octants(center: Vec3iLike, point: Vec3iLike) -> List[ivec3]:
        """Generates octants for a point around a center"""
        x0, y0, z0 = center
        x, y, z = point
        dx, dy, dz = x - x0, y - y0, z - z0

        octants: List[ivec3] = [
            ivec3(x0 + e.x + dx, y0 + e.y + dy, z0 + e.z + dz),
            ivec3(x0       - dx, y0 + e.y + dy, z0 + e.z + dz),
            ivec3(x0 + e.x + dx, y0       - dy, z0 + e.z + dz),
            ivec3(x0       - dx, y0       - dy, z0 + e.z + dz),
            ivec3(x0 + e.x + dx, y0 + e.y + dy, z0       - dz),
            ivec3(x0       - dx, y0 + e.y + dy, z0       - dz),
            ivec3(x0 + e.x + dx, y0       - dy, z0       - dz),
            ivec3(x0       - dx, y0       - dy, z0       - dz),
        ]

        return octants

    # Extract the x, y, and z coordinates of the center point
    x0, y0, z0 = center
    # Extract the radii of the ellipsoid along the x, y, and z axes
    rx, ry, rz = ((diameters) // 2) + (1 - e)

    solid_points: np.ndarray[Any, np.dtype[bool]] = np.zeros((rx + 2, ry + 2, rz + 2), dtype=bool)

    # Loop over all points within the bounding box of the ellipsoid
    for x, y, z in itertools.product(*(range(n) for n in solid_points.shape)):

        # Compute the ellipsoid equation for the current point
        e_val: float = (x**2 / rx**2) + (y**2 / ry**2) + (z**2 / rz**2)
        # Check if it is in-line with the center point
        in_line_with_center: bool = are_points_in_line(
            center, (x + x0, y + y0, z + z0)
        )

        # If the point satisfies the ellipsoid equation
        if e_val <= 1 and (not in_line_with_center or e_val < 1):
            # If it should be hollow, add the point to the point array
            if hollow: solid_points[x, y, z] = True
            # Otherwise, yield it for all octants
            else:
                yield from generate_octants(
                    center, ivec3(x + x0, y + y0, z + z0)
                )

    # If the ellipsoid should be hollow
    if hollow:
        # Iterate through every point in the array, except the outer faces
        for x, y, z in itertools.product(*(range(n-1) for n in solid_points.shape)):

            # A point is considered part of the "shell" if it meets the following conditions: (Thanks to @Jandhi#5234 on discord)
            # - It is part of the solid ellipsoid
            # - At least one of it's adjacent points isn't (we only have to check 3/6 because of octants)
            shell: bool = solid_points[x, y, z] and (
                not solid_points[x + 1, y, z]
                or not solid_points[x, y + 1, z]
                or not solid_points[x, y, z + 1]
            )

            # If a point is part of the shell, yield it for all octants
            if shell: yield from generate_octants(center, ivec3(x + x0, y + y0, z + z0))


def fittingEllipsoid(corner1: Vec3iLike, corner2: Vec3iLike, hollow: bool = False) -> Generator[ivec3, Any, None]:
    """Yields the points of the largest ellipsoid that fits between <corner1> and <corner2>."""
    corner1_, corner2_ = orderedCorners3D(corner1, corner2)
    diameters: ivec3 = corner2_ - corner1_ + 1
    center: ivec3 = (corner1_ + corner2_) // 2
    return ellipsoid(center, diameters, hollow)


def sphere(center: Vec3iLike, diameter: int, hollow: bool = False) -> Generator[ivec3, Any, None]:
    """Yields the points of a sphere centered on <center> with diameter <diameter>.\n
    If <diameter> is even, <center> will be the lower center point in every axis."""
    return ellipsoid(center, (diameter, diameter, diameter), hollow)


def fittingSphere(corner1: Vec3iLike, corner2: Vec3iLike, hollow: bool = False) -> Generator[ivec3, Any, None]:
    """Yields the points of the largest sphere that fits between <corner1> and <corner2>.\n
    The circle will be centered in the non-minimum axes."""
    corner1_, corner2_ = orderedCorners3D(corner1, corner2)
    diameter: int = min(corner2_ - corner1_) + 1
    center: ivec3 = (corner1_ + corner2_) // 2
    return sphere(center, diameter, hollow)


def _boundedNeighborsFromVectors2D(point: ivec2, bounding_rect: Rect, vectors: Iterable[ivec2], stride: int = 1) -> Generator[ivec2, Any, None]:
    """Generate neighboring vectors within a bounding rect in the directions of vectors."""
    for vector in vectors:
        candidate: ivec2 = point + stride * vector
        if bounding_rect.contains(candidate):
            yield candidate


def neighbors2D(point: Vec2iLike, boundingRect: Rect, diagonal: bool = False, stride: int = 1) -> Generator[ivec2, Any, None]:
    """Yields the neighbors of [point] within [bounding_rect].\n
    Useful for pathfinding."""
    point = ivec2(*point)
    vectors: FrozenSet[ivec2] = CARDINALS_AND_DIAGONALS_2D if diagonal else CARDINALS_2D
    return _boundedNeighborsFromVectors2D(point, boundingRect, vectors, stride)


def _boundedNeighborsFromVectors3D(point: ivec3, bounding_box: Box, vectors: Iterable[ivec3], stride: int = 1) -> Generator[ivec3, Any, None]:
    """Generate neighboring vectors within a bounding box in the directions of vectors."""
    for vector in vectors:
        candidate: ivec3 = point + stride * vector
        if bounding_box.contains(candidate):
            yield candidate


def neighbors3D(point: Vec3iLike, boundingBox: Box, diagonal: bool = False, stride: int = 1) -> Generator[ivec3, Any, None]:
    """Yields the neighbors of [point] within [bounding_box].\n
    Useful for pathfinding."""
    point = ivec3(*point)
    vectors: FrozenSet[ivec3] = DIRECTIONS_AND_ALL_DIAGONALS_3D if diagonal else DIRECTIONS_3D
    return _boundedNeighborsFromVectors3D(point, boundingBox, vectors, stride)
