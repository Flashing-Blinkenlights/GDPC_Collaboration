"""Provides the `Editor` class, which provides high-level functions to interact with the Minecraft
world through the GDMC HTTP interface"""


from typing import Dict, Union, Optional, List, Iterable
from contextlib import contextmanager
from copy import copy, deepcopy
from concurrent import futures

import numpy as np
from glm import ivec3
from termcolor import colored

from .utility import eprint, eagerAll, OrderedByLookupDict
from .vector_tools import Rect, Box, addY, dropY
from .transform import Transform, TransformLike, toTransform
from .block import Block
from . import lookup
from .interface import Interface
from .world_slice import WorldSlice


class Editor:
    """Provides a high-level functions to interact with the Minecraft world through the GDMC HTTP
    interface.

    Stores various settings, resources, buffers and caches related to block placement, and a
    transform that defines a local coordinate system.
    """

    def __init__(
        self,
        transformLike: Optional[TransformLike] = None,
        buffering             = False,
        bufferLimit           = 1024,
        caching               = False,
        cacheLimit            = 8192,
        multithreading        = False,
        multithreadingWorkers = 1,
        interface: Optional[Interface] = None
    ):
        """Constructs an Editor instance with the specified transform and settings"""
        self._interface = Interface() if interface is None else interface

        self._transform = Transform() if transformLike is None else toTransform(transformLike)

        self._buffering = buffering
        self._bufferLimit = bufferLimit
        self._buffer: Dict[ivec3,Block] = {}
        self._commandBuffer: List[str] = []

        self._caching = caching
        self._cache = OrderedByLookupDict[ivec3,Block](cacheLimit)

        self._multithreading = False
        self._multithreadingWorkers = multithreadingWorkers
        self.multithreading = multithreading # The property setter initializes the multithreading system.
        self._bufferFlushFutures: List[futures.Future] = []

        self._doBlockUpdates = True
        self._bufferDoBlockUpdates = True

        self._worldSlice: Optional[WorldSlice] = None
        self._worldSliceDecay: Optional[np.ndarray] = None


    def __del__(self):
        """Cleans up this Editor instance"""
        # awaits any pending buffer flush futures and shuts down the buffer flush executor
        self.multithreading = False # awaits any pending buffer flush futures and shuts down the buffer flush executor
        # Flush any remaining blocks in the buffer.
        # This is purposefully done *after* disabling multithreading! This __del__ may be called at
        # interpreter shutdown, and it appears that scheduling a new future at that point fails with
        # "RuntimeError: cannot schedule new futures after shutdown" even if the executor has not
        # actually shut down yet. For safety, the last buffer flush must be done on the main thread.
        self.sendBufferedBlocks()


    @property
    def interface(self):
        """This editor's underlying interface"""
        return self._interface

    @interface.setter
    def interface(self, value: Interface):
        self.sendBufferedBlocks()
        self.awaitBufferFlushes()
        self._interface = value

    @property
    def transform(self):
        """This editor's local coordinate transform (used for block placement and retrieval)"""
        return self._transform

    @transform.setter
    def transform(self, value: Union[Transform, ivec3]):
        self._transform = toTransform(value)

    @property
    def buffering(self) -> bool:
        """Whether block placement buffering is enabled"""
        return self._buffering

    @buffering.setter
    def buffering(self, value: bool):
        if self.buffering and not value:
            self.sendBufferedBlocks()
        self._buffering = value

    @property
    def bufferLimit(self) -> int:
        """Size of the block buffer"""
        return self._bufferLimit

    @bufferLimit.setter
    def bufferLimit(self, value: int):
        self._bufferLimit = value
        if len(self._buffer) >= self.bufferLimit:
            self.sendBufferedBlocks()

    @property
    def caching(self):
        """Whether caching placed and retrieved blocks is enabled"""
        return self._caching

    @caching.setter
    def caching(self, value: bool):
        self._caching = value

    @property
    def cacheLimit(self):
        """Size of the block cache"""
        return self._cache.maxSize

    @cacheLimit.setter
    def cacheLimit(self, value: int):
        self._cache.maxSize = value

    @property
    def multithreading(self):
        """Whether multithreaded buffer flushing is enabled"""
        return self._multithreading

    @multithreading.setter
    def multithreading(self, value: bool):
        if not self._multithreading and value:
            self._bufferFlushExecutor = futures.ThreadPoolExecutor(self._multithreadingWorkers)
            if self._multithreadingWorkers > 1:
                eprint(colored(color="yellow", text=\
                    "WARNING: An editor has been set to use multithreaded buffer flushing with more\n"
                    "than one worker thread.\n"
                    "The editor can no longer guarantee that blocks will be placed in the same order\n"
                    "as they were sent. If buffering or caching is used, this can also cause the\n"
                    "caches to become inconsistent with the actual world.\n"
                    "Multithreading with more than one worker thread can speed up block placement on\n"
                    "some machines, which can be nice during development, but it is NOT RECOMMENDED\n"
                    "for production code."
                ))
        elif self._multithreading and not value:
            self._bufferFlushExecutor.shutdown(wait=True)
            del self._bufferFlushExecutor
        self._multithreading = value

    @property
    def multithreadingWorkers(self):
        """The amount of buffer flush worker threads."""
        return self._multithreadingWorkers

    @multithreadingWorkers.setter
    def multithreadingWorkers(self, value: int):
        restartExecutor = self.multithreading and self._multithreadingWorkers != value
        self._multithreadingWorkers = value
        if restartExecutor:
            self.multithreading = False
            self.multithreading = True

    # TODO: Add support for the other block placement flags?
    # https://github.com/nilsgawlik/gdmc_http_interface/wiki/Interface-Endpoints
    @property
    def doBlockUpdates(self):
        return self._doBlockUpdates

    @doBlockUpdates.setter
    def doBlockUpdates(self, value: bool):
        self._doBlockUpdates = value

    @property
    def worldSlice(self):
        """The cached WorldSlice"""
        return self._worldSlice

    @property
    def worldSliceDecay(self):
        """3D boolean array indicating whether the block at the specified position in the cached
        worldSlice is still valid.\n
        Do not edit the returned array.\n
        Note that the lowest Y-layer is at [:,0,:], despite Minecraft's negative Y coordinates."""
        return self._worldSliceDecay


    def runCommand(self, command: str, syncWithBuffer=False):
        """Executes one or multiple Minecraft commands (separated by newlines).\n
        The leading "/" must be omitted.\n
        If buffering is enabled and <syncWithBuffer>=True, the command is deferred until after the
        next buffer flush."""
        if self.buffering and syncWithBuffer:
            self._commandBuffer.append(command)
            return
        self._interface.runCommand(command)


    def getBuildArea(self) -> Box:
        """Returns the build area that was specified by /setbuildarea in-game.\n
        The build area is always in **global coordinates**; self.transform is ignored."""
        success, result = self.interface.getBuildArea()
        if not success:
            raise RuntimeError(colored(color="red", text=\
                "ERROR: Failed to get build area!\n"
                "Make sure to set the build area with /setbuildarea in-game.\n"
                "For example: /setbuildarea ~0 0 ~0 ~128 255 ~128"
            ))
        return result


    def setBuildArea(self, buildArea: Box):
        """Sets the build area to [box], and returns it.\n
        The build area must be given in **global coordinates**; self.transform is ignored."""
        self.runCommand(f"setbuildarea {buildArea.begin.x} {buildArea.begin.y} {buildArea.begin.z} {buildArea.end.x} {buildArea.end.y} {buildArea.end.z}")
        return self.getBuildArea()


    def getBlock(self, position: ivec3, getBlockStates=True, getBlockData=False):
        """Returns the block at [position].\n
        <position> is interpreted as local to the coordinate system defined by self.transform.
        The returned block's orientation is also from the perspective of self.transform.\n
        If the given coordinates are invalid, returns Block("minecraft:void_air")."""
        block = self.getBlockGlobal(self.transform * position, getBlockStates, getBlockData)
        invTransform = ~self.transform
        block.transform(invTransform.rotation, invTransform.flip)
        return block


    def getBlockGlobal(self, position: ivec3, getBlockStates=True, getBlockData=False):
        """Returns the block at [position], ignoring self.transform.\n
        If the given coordinates are invalid, returns Block("minecraft:void_air")."""
        if self.caching:
            try:
                return copy(self._cache[position])
            except TypeError:
                pass

        if self.buffering:
            block = self._buffer.get(position, None)
            if block is not None:
                return block

        if (
            self._worldSlice is not None and
            self._worldSlice.rect.contains(dropY(position)) and
            not self._worldSliceDecay[tuple(position - addY(self._worldSlice.rect.offset, lookup.BUILD_Y_MIN))]
        ):
            block = Block.fromBlockCompound(self._worldSlice.getBlockCompoundAt(position))
        else:
            block = self.interface.getBlocks(position, includeState=getBlockStates, includeData=getBlockData)[0][1]

        if self.caching:
            self._cache[position] = copy(block)

        return block


    def placeBlock(
        self,
        position:       Union[ivec3, Iterable[ivec3]],
        block:          Block,
        replace:        Optional[Union[str, List[str]]] = None,
        doBlockUpdates: Optional[bool] = None
    ):
        """Places <block> at <position>.\n
        <position> is interpreted as local to the coordinate system defined by self.transform.\n
        If <position> is iterable (e.g. a list), <block> is placed at all positions.
        This is slightly more efficient than calling this method in a loop.\n
        If <block>.name is a list, names are sampled randomly.\n
        Returns whether the placement succeeded fully."""
        return self.placeBlockGlobal(
            self.transform * position if isinstance(position, ivec3) else (self.transform * pos for pos in position),
            block.transformed(self.transform.rotation, self.transform.flip),
            replace,
            doBlockUpdates
        )


    def placeBlockGlobal(
        self,
        position:       Union[ivec3, Iterable[ivec3]],
        block:          Block,
        replace:        Optional[Union[str, Iterable[str]]] = None,
        doBlockUpdates: Optional[bool] = None
    ):
        """Places <block> at <position>, ignoring self.transform.\n
        If <position> is iterable (e.g. a list), <block> is placed at all positions.
        In this case, buffering is temporarily enabled for better performance.\n
        If <block>.name is a list, names are sampled randomly.\n
        Returns whether the placement succeeded fully."""

        if isinstance(position, ivec3):
            return self._placeSingleBlockGlobal(position, block, replace, doBlockUpdates)

        oldBuffering = self.buffering
        self.buffering = True
        success = eagerAll(self._placeSingleBlockGlobal(pos, block, replace, doBlockUpdates) for pos in position)
        self.buffering = oldBuffering
        return success


    def _placeSingleBlockGlobal(
        self,
        position:       ivec3,
        block:          Block,
        replace:        Optional[Union[str, Iterable[str]]] = None,
        doBlockUpdates: Optional[bool] = None
    ):
        """Places <block> at <position>, ignoring self.transform.\n
        If <block>.name is a list, names are sampled randomly.\n
        Returns whether the placement succeeded fully."""

        # Check replace condition
        if replace is not None:
            if isinstance(replace, str):
                replace = [replace]
            if self.getBlockGlobal(position) not in replace:
                return True

        # Select block from palette
        block = block.chooseId()

        # Support for "no placement" in block palettes
        if not block.id:
            return True

        if (self.caching and block.id in self.getBlockGlobal(position)): # TODO: this is very error-prone! "stone" is in "stone_stairs". Also, we may want to change only block state or nbt data.
            return True

        if self._buffering:
            success = self._placeSingleBlockGlobalBuffered(position, block, doBlockUpdates)
        else:
            success = self._placeSingleBlockGlobalDirect(position, block, doBlockUpdates)

        if not success:
            return False

        if self.caching:
            self._cache[position] = block

        if self._worldSlice is not None and self._worldSlice.rect.contains(dropY(position)):
            self._worldSliceDecay[tuple(position - addY(self._worldSlice.rect.offset, lookup.BUILD_Y_MIN))] = True

        return True


    def _placeSingleBlockGlobalDirect(self, position: ivec3, block: Block, doBlockUpdates: Optional[bool]):
        """Place a single block in the world directly.\n
        Returns whether the placement succeeded."""
        if doBlockUpdates is None: doBlockUpdates = self.doBlockUpdates

        result = self.interface.placeBlocks([(position, block)], doBlockUpdates=doBlockUpdates)
        if not result[0].isnumeric():
            eprint(colored(color="yellow", text=f"Warning: Server returned error upon placing block:\n\t{result}"))
            return False
        return True


    def _placeSingleBlockGlobalBuffered(self, position: ivec3, block: Block, doBlockUpdates: Optional[bool]):
        """Place a block in the buffer and send once limit is exceeded.\n
        Returns whether placement succeeded."""
        if doBlockUpdates is None: doBlockUpdates = self.doBlockUpdates

        if doBlockUpdates != self._bufferDoBlockUpdates:
            self.sendBufferedBlocks()
            self._bufferDoBlockUpdates = doBlockUpdates

        elif len(self._buffer) >= self.bufferLimit:
            self.sendBufferedBlocks()

        self._buffer[position] = block
        return True


    def sendBufferedBlocks(self, retries = 5):
        """Flushes the block placement buffer.\n
        If multithreaded buffer flushing is enabled, the threads can be awaited with
        awaitBufferFlushes()."""

        def flush(blockBuffer: Dict[ivec3, Block], commandBuffer: List[str]):
            # Flush block buffer
            if blockBuffer:
                response = self.interface.placeBlocks(blockBuffer.items(), doBlockUpdates=self._bufferDoBlockUpdates, retries=retries)
                blockBuffer.clear()

                for line in response:
                    if not line.isnumeric():
                        eprint(colored(color="yellow", text=f"Warning: Server returned error upon placing buffered block:\n\t{line}"))

            # Flush command buffer
            if commandBuffer:
                response = self._interface.runCommand("\n".join(commandBuffer))
                commandBuffer.clear()

                for line in response:
                    if not line.isnumeric():
                        eprint(colored(color="yellow", text=f"Warning: Server returned error upon sending buffered command:\n\t{line}"))

        if self._multithreading:
            # Clean up finished buffer flush futures
            self._bufferFlushFutures = [
                future for future in self._bufferFlushFutures if not future.done()
            ]

            # Shallow copies are good enough here
            blockBufferCopy   = self._buffer
            commandBufferCopy = self._commandBuffer
            def task():
                flush(blockBufferCopy, commandBufferCopy)

            # Submit the task
            future = self._bufferFlushExecutor.submit(task)
            self._bufferFlushFutures.append(future)

            # Empty the buffers (the thread has copies of the references)
            self._buffer = {}
            self._commandBuffer = []

        else: # No multithreading
            flush(self._buffer, self._commandBuffer)


    def awaitBufferFlushes(self, timeout: Optional[float] = None):
        """Awaits all pending buffer flushes.\n
        If [timeout] is not None, waits for at most [timeout] seconds.\n
        Does nothing if no buffer flushes have occured while multithreaded buffer flushing was
        enabled."""
        self._bufferFlushFutures = futures.wait(self._bufferFlushFutures, timeout).not_done


    def loadWorldSlice(self, rect: Optional[Rect]=None, cache=False):
        """Loads the world slice for the given XZ-rectangle.\n
        The rectangle must be given in **global coordinates**; self.transform is ignored.\n
        If <rect> is None, the world slice of the current build area is loaded.\n
        If <cache>=True, the loaded worldSlice is cached in this editor. It can then be accessed
        through .worldSlice.
        If a world slice was already cached, it is replaced.
        The cached world slice is used for faster block retrieval. Note that the editor assumes
        that nothing besides itself changes the given area of the world. If the given world area is
        changed other than through this editor, call .updateWorldSlice() to update the cached world
        slice."""
        if rect is None:
            rect = self.getBuildArea()
        worldSlice = WorldSlice(self.interface, rect)
        if cache:
            self._worldSlice      = worldSlice
            self._worldSliceDecay = np.zeros(addY(self._worldSlice.rect.size, lookup.BUILD_HEIGHT), dtype=np.bool)
        return worldSlice


    def updateWorldSlice(self):
        """Updates the cached world slice."""
        if self._worldSlice is None:
            raise RuntimeError("No world slice is cached. Call .loadWorldSlice() with cache=True first.")
        return self.loadWorldSlice(self._worldSlice.rect)


    def getMinecraftVersion(self):
        """Returns the Minecraft version as a string."""
        return self.interface.getVersion()


    def isConnected(self):
        """Returns whether the underlying Interface is connected to an active host."""
        return self.interface.isConnected()


    @contextmanager
    def pushTransform(self, transformLike: Optional[TransformLike] = None):
        """Creates a context that reverts all changes to self.transform on exit.
        If <transformLine> is not None, it is pushed to self.transform on enter.

        Can be used to create a local coordinate system on top of the current local coordinate
        system.

        Not to be confused with Transform.push()!"""

        originalTransform = deepcopy(self.transform)
        if transformLike is not None:
            self.transform @= toTransform(transformLike)
        try:
            yield
        finally:
            self.transform = originalTransform