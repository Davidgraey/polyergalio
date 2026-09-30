import asyncio
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

log = logging.getLogger(__name__)

DEFAULT_PORT = int(os.environ.get("POLYERGALIO_PORT", 31352))
DEFAULT_ROSTER = Path(os.environ.get("POLYERGALIO_ROSTER", "nodes.toml"))
BROADCAST = "255.255.255.255"
MAGIC = "polyergalio"
PACKET_LIMIT = 512


@dataclass
class NodeRecord:
    """A node as seen on the network or listed in the roster."""
    node_id: str
    host: str
    port: int = DEFAULT_PORT
    last_seen: float = 0.0
    busy: bool = False

    @property
    def address(self) -> tuple[str, int]:
        return self.host, self.port


def parse_address(address: Union[str, tuple], default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    """Normalize "host", "host:port" or (host, port) to (host, port)."""
    if isinstance(address, str):
        host, separator, port = address.rpartition(":")
        if not separator:
            return address, default_port
        return host, int(port)
    host, port = address
    return str(host), int(port)


def quote(value: str) -> str:
    return json.dumps(value)


def parse_value(text: str):
    if text[:1] in "\"'":
        return json.loads(text) if text[0] == '"' else text[1:-1]
    try:
        return int(text)
    except ValueError:
        return float(text)


def strip_comment(line: str) -> str:
    quote_mark = None
    for index, character in enumerate(line):
        if quote_mark:
            quote_mark = None if character == quote_mark and line[index - 1] != "\\" else quote_mark
        elif character in "\"'":
            quote_mark = character
        elif character == "#":
            return line[:index].strip()
    return line.strip()


class Roster:
    """
    TOML file listing the nodes the orchestrator may use; the source of truth for addresses.

    The file is an array of ``[[node]]`` tables with ``node_id``, ``host``, ``port`` and
    ``last_seen`` keys. Only that subset of TOML is read and written.

    Parameters
    ----------
    path : roster file, created on the first write
    """

    def __init__(self, path: Union[str, Path] = DEFAULT_ROSTER):
        self.path = Path(path)

    def load(self) -> list[NodeRecord]:
        if not self.path.exists():
            return []
        tables: list[dict] = []
        for line in self.path.read_text().splitlines():
            line = strip_comment(line)
            if not line:
                continue
            if line == "[[node]]":
                tables.append({})
            elif "=" in line and tables:
                key, _, value = line.partition("=")
                tables[-1][key.strip()] = parse_value(value.strip())
            else:
                raise ValueError(f"unsupported roster line: {line!r}")
        return [
            NodeRecord(
                node_id=str(table.get("node_id", table["host"])),
                host=str(table["host"]),
                port=int(table.get("port", DEFAULT_PORT)),
                last_seen=float(table.get("last_seen", 0.0)),
            )
            for table in tables
        ]

    def save(self, records: Iterable[NodeRecord]) -> None:
        blocks = [
            f"[[node]]\nnode_id = {quote(record.node_id)}\nhost = {quote(record.host)}\n"
            f"port = {record.port}\nlast_seen = {record.last_seen}\n"
            for record in records
        ]
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text("\n".join(blocks))
        os.replace(temporary, self.path)

    def update(self, found: Iterable[NodeRecord]) -> list[NodeRecord]:
        """Merge discovered nodes into the file, matching by node_id then by address; all records."""
        records = self.load()
        for new in found:
            match = next((r for r in records if r.node_id == new.node_id), None)
            match = match or next((r for r in records if r.address == new.address), None)
            if match is None:
                records.append(NodeRecord(new.node_id, new.host, new.port, new.last_seen))
            else:
                match.node_id, match.host, match.port, match.last_seen = new.node_id, new.host, new.port, new.last_seen
        self.save(records)
        return records

    def remove(self, node_id: str) -> None:
        self.save(record for record in self.load() if record.node_id != node_id)

    def addresses(self) -> list[tuple[str, int]]:
        return [record.address for record in self.load()]


def beacon_socket(port: int, broadcast: bool = False) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if broadcast:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", port))
    sock.setblocking(False)
    return sock


def decode(data: bytes, kind: str) -> Optional[dict]:
    if len(data) > PACKET_LIMIT:
        return None
    try:
        message = json.loads(data)
    except ValueError:
        return None
    if not isinstance(message, dict) or message.get("magic") != MAGIC or message.get("type") != kind:
        return None
    return message


class Responder(asyncio.DatagramProtocol):
    """Answers discovery probes with this node's id, TCP port and busy flag."""

    def __init__(self, node_id: str, port: int, busy: Callable[[], bool]):
        self.node_id = node_id
        self.port = port
        self.busy = busy
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, address):
        if decode(data, "discover") is None:
            return
        reply = {"magic": MAGIC, "type": "announce", "node_id": self.node_id,
                 "port": self.port, "busy": bool(self.busy())}
        self.transport.sendto(json.dumps(reply).encode(), address)


async def announce(node_id: str, port: int, busy: Callable[[], bool] = lambda: False,
                   beacon_port: int = DEFAULT_PORT) -> asyncio.DatagramTransport:
    """Start answering probes on the beacon port; close the returned transport to stop."""
    transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
        lambda: Responder(node_id, port, busy), sock=beacon_socket(beacon_port))
    return transport


class Collector(asyncio.DatagramProtocol):
    def __init__(self):
        self.found: dict[str, NodeRecord] = {}

    def datagram_received(self, data, address):
        message = decode(data, "announce")
        if message is None:
            return
        try:
            record = NodeRecord(str(message["node_id"])[:128], address[0], int(message["port"]),
                                time.time(), bool(message.get("busy", False)))
        except (KeyError, ValueError, TypeError):
            return
        if 0 < record.port < 65536:
            self.found[record.node_id] = record


async def find_nodes(timeout: float = 2.0, beacon_port: int = DEFAULT_PORT,
                     targets: Iterable[str] = (BROADCAST,)) -> list[NodeRecord]:
    """
    Probe the network and collect the nodes that answer.

    Parameters
    ----------
    timeout : seconds to listen for answers; the probe is sent at the start and again halfway
    beacon_port : UDP port nodes listen on
    targets : broadcast or unicast addresses to probe, as "host" or "host:beacon_port"
    """
    loop = asyncio.get_running_loop()
    collector = Collector()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: collector, sock=beacon_socket(0, broadcast=True))
    probe = json.dumps({"magic": MAGIC, "type": "discover"}).encode()
    try:
        for _ in range(2):
            for target in targets:
                try:
                    transport.sendto(probe, parse_address(target, beacon_port))
                except OSError as error:
                    log.warning("probe to %s failed: %s", target, error)
            await asyncio.sleep(timeout / 2)
    finally:
        transport.close()
    return list(collector.found.values())
