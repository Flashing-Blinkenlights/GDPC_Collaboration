"""Provides various utilities that require an Editor."""


from typing import Optional, Iterable, Set, Tuple, Union
import random

import numpy as np
from glm import ivec2, ivec3

from .vector_tools import Box, neighbors3D
from .block import Block
from .block_state_tools import facingToRotation, facingToVector
from .minecraft_tools import getObtrusiveness, lecternBlock, positionToInventoryIndex, signBlock
from . import lookup
from .editor import Editor


def centerBuildAreaOnPlayer(editor: Editor, size: ivec3):
    """Sets <editor>'s build area to a box of <size> centered on the player, and returns it.\n
    The build area is always in **global coordinates**; <editor>.transform is ignored."""
    # -1 to correct for offset from player position
    radius = (size - 1) // 2
    editor.runCommand(
        "execute at @p run setbuildarea "
        f"~{-radius.x} ~{-radius.y} ~{-radius.z} ~{radius.x} ~{radius.y} ~{radius.z}")
    return editor.getBuildArea()


def flood_search_3D(
    editor: Editor,
    origin: ivec3,
    boundingBox: Box,
    search_block_ids: Iterable[str],
    diagonal=False,
    depth=256
):
    """Return a list of coordinates with blocks that fulfill the search.\n
    Activating caching is *highly* recommended."""
    def flood_search_3D_recursive(point: ivec3, result: Set[ivec3], visited: Set[ivec3], depth_: int):
        if point in visited:
            return

        visited.add(point)

        if editor.getBlock(point) not in search_block_ids:
            return

        result.add(point)

        for neighbor in neighbors3D(point, boundingBox, diagonal):
            flood_search_3D_recursive(neighbor, result, visited, depth_ - 1)

    result:  Set[ivec3] = set()
    visited: Set[ivec3] = set()
    flood_search_3D_recursive(origin, result, visited, depth)
    return result


def placeSign(
    editor: Editor,
    position: ivec3,
    wood="oak", wall=False,
    facing: Optional[str] = None, rotation: Optional[Union[str, int]] = None,
    text1="", text2="", text3="", text4="", color=""
):
    """Places a sign with the specified properties.\n
    If <wall> is True, <facing> is used. Otherwise, <rotation> is used.
    If the used property is None, a least obstructed direction will be used."""
    if wall and facing is None:
        facing = random.choice(getOptimalFacingDirection(editor, position))
    elif not wall and rotation is None:
        rotation = facingToRotation(random.choice(getOptimalFacingDirection(editor, position)))
    editor.placeBlock(position, signBlock(wood, wall, facing, rotation, text1, text2, text3, text4, color))


def placeLectern(editor: Editor, position: ivec3, facing: Optional[str] = None, bookData: Optional[str] = None, page: int = 0):
    """Place a lectern with the specified properties.\n
    If <facing> is None, a least obstructed facing direction will be used."""
    if facing is None:
        facing = random.choice(getOptimalFacingDirection(editor, position))
    editor.placeBlock(position, lecternBlock(facing, bookData, page))


def placeContainerBlock(
    editor: Editor,
    position: ivec3,
    block: Block = Block("minecraft:chest"),
    items: Optional[Iterable[Union[Tuple[ivec2, str], Tuple[ivec2, str, int]]]] = None,
    replace=True
):
    """Place a container block with the specified items in the world.\n
    <items> should be a sequence of (position, item, [amount,])-tuples."""
    if block.id not in lookup.CONTAINER_BLOCK_TO_INVENTORY_SIZE:
        raise ValueError(f'"{block}" is not a known container block. Make sure you are using its namespaced ID.')
    inventorySize = lookup.CONTAINER_BLOCK_TO_INVENTORY_SIZE[block]

    if not replace and editor.getBlock(position).id != block.id:
        return

    editor.placeBlock(position, block)

    if items is None:
        return

    for item in items:
        index = positionToInventoryIndex(item[0], inventorySize)
        if len(item) == 3:
            item = list(item)
            item.append(1)
        globalPosition = editor.transform * position
        editor.runCommand(f"replaceitem block {' '.join(globalPosition)} container.{index} {item[2]} {item[3]}", syncWithBuffer=True)


def setContainerItem(editor: Editor, position: ivec3, itemPosition: ivec2, item: str, amount: int = 1):
    """Sets the item at <itemPosition> in the container block at <position> to the item with id <item>."""
    globalPosition = editor.transform * position

    block = editor.getBlockGlobal(globalPosition)
    if block.id not in lookup.CONTAINER_BLOCK_TO_INVENTORY_SIZE:
        raise ValueError(f'The block at ({",".join(position)}) is "{block}", which is not a known container block.')
    inventorySize = lookup.CONTAINER_BLOCK_TO_INVENTORY_SIZE[block]

    index = positionToInventoryIndex(itemPosition, inventorySize)
    editor.runCommand(f"replaceitem block {' '.join(globalPosition)} container.{index} {item} {amount}", syncWithBuffer=True)


def getOptimalFacingDirection(editor: Editor, pos: ivec3):
    """Returns the least obstructed directions to have something facing (a "facing" block state value).\n
    Ranks directions by obtrusiveness first, and by obtrusiveness of the opposite direction second."""
    directions = ["north", "east", "south", "west"]
    obtrusivenesses = np.array([
        getObtrusiveness(editor.getBlock(pos + facingToVector(direction)))
        for direction in directions
    ])
    candidates              = np.nonzero(obtrusivenesses == np.min(obtrusivenesses))[0]
    oppositeObtrusivenesses = obtrusivenesses[(candidates + 2) % 4]
    winners                 = candidates[oppositeObtrusivenesses == np.max(oppositeObtrusivenesses)]
    return [directions[winner] for winner in winners]