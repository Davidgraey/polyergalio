import asyncio
import hashlib
import hmac
import io
import pickle
import struct
from typing import Any

LENGTH = struct.Struct(">Q")
HANDSHAKE_LIMIT = 4096
SKELETON_LIMIT = 1 << 26
BUFFER_LIMIT = 1 << 34
BUFFER_COUNT_LIMIT = 1 << 16
CHUNK_BYTES = 1 << 20

ALLOWED_GLOBALS = {
    "builtins": {"dict", "list", "tuple", "set", "frozenset", "str", "int", "float", "bool", "bytes", "bytearray",
                 "complex", "slice", "range"},
    "collections": {"OrderedDict"},
    "numpy": {"ndarray", "dtype"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy.core.numeric": {"_frombuffer"},
    "numpy._core.numeric": {"_frombuffer"},
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler limited to builtin containers, numbers, and NUMPY"""

    def find_class(self, module: str, name: str):
        if name in ALLOWED_GLOBALS.get(module, ()):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"{module}.{name} is not allowed on the wire")


def sign(secret: bytes, nonce: str) -> str:
    """HMAC digest proving knowledge of the shared secret."""
    return hmac.new(secret, nonce.encode(), hashlib.sha256).hexdigest()


async def send_frame(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Write one length-prefixed frame."""
    writer.write(LENGTH.pack(len(data)) + data)
    await writer.drain()


async def receive_frame(reader: asyncio.StreamReader, limit: int = HANDSHAKE_LIMIT) -> bytes:
    """Read one length-prefixed frame no larger than limit."""
    (length,) = LENGTH.unpack(await reader.readexactly(LENGTH.size))
    if length > limit:
        raise ValueError(f"frame of {length} bytes exceeds limit of {limit}")
    return await reader.readexactly(length)


async def send_message(writer: asyncio.StreamWriter, message: Any) -> None:
    """
    Send a picklable message, streaming array buffers separately from the pickled skeleton.
    """
    buffers = []
    skeleton = pickle.dumps(message, protocol=5, buffer_callback=buffers.append)
    writer.write(LENGTH.pack(len(buffers)))
    await send_frame(writer, skeleton)
    for buffer in buffers:
        view = buffer.raw()
        writer.write(LENGTH.pack(view.nbytes))
        for start in range(0, view.nbytes, CHUNK_BYTES):
            writer.write(view[start:start + CHUNK_BYTES])
            await writer.drain()
    await writer.drain()


async def receive_buffer(reader: asyncio.StreamReader, length: int) -> bytearray:
    """Read bytes into one preallocated, writable buffer."""
    buffer = bytearray(length)
    view = memoryview(buffer)
    offset = 0
    while offset < length:
        chunk = await reader.read(min(CHUNK_BYTES, length - offset))
        if not chunk:
            raise asyncio.IncompleteReadError(bytes(view[:offset]), length)
        view[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    return buffer


async def receive_message(reader: asyncio.StreamReader) -> Any:
    """Receive a message written by send_message, unpickled with RestrictedUnpickler."""
    (count,) = LENGTH.unpack(await reader.readexactly(LENGTH.size))
    if count > BUFFER_COUNT_LIMIT:
        raise ValueError(f"{count} buffers exceeds limit of {BUFFER_COUNT_LIMIT}")
    skeleton = await receive_frame(reader, SKELETON_LIMIT)
    buffers = []
    for _ in range(count):
        (length,) = LENGTH.unpack(await reader.readexactly(LENGTH.size))
        if length > BUFFER_LIMIT:
            raise ValueError(f"buffer of {length} bytes exceeds limit of {BUFFER_LIMIT}")
        buffers.append(await receive_buffer(reader, length))
    return RestrictedUnpickler(io.BytesIO(skeleton), buffers=buffers).load()
