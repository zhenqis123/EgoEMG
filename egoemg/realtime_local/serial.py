from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Callable

import numpy as np


def _int24_be_signed(b0: int, b1: int, b2: int) -> int:
    value = (b0 << 16) | (b1 << 8) | b2
    if value & 0x800000:
        value -= 1 << 24
    return value


def _load_collect_protocol() -> tuple[bytes | None, int | None, int | None, int | None]:
    try:
        from collect import HEADER, PKT_LEN, TYPE_AA, TYPE_BB  # type: ignore

        return bytes(HEADER), int(PKT_LEN), int(TYPE_AA), int(TYPE_BB)
    except Exception:
        pass

    for root in [Path.cwd(), *Path.cwd().parents]:
        collect_py = root / "collect.py"
        if collect_py.exists():
            parsed = _parse_collect_protocol(collect_py)
            if parsed[0] is not None:
                return parsed
    return None, None, None, None


def _parse_collect_protocol(path: Path) -> tuple[bytes | None, int | None, int | None, int | None]:
    values: dict[str, object] = {}
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not names:
            continue
        name = names[0]
        if name not in {"HEADER", "PKT_LEN", "TYPE_AA", "TYPE_BB"}:
            continue
        if name == "HEADER" and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "bytes" and node.value.args:
                arg = node.value.args[0]
                if isinstance(arg, ast.List):
                    values[name] = bytes(
                        int(elt.value)
                        for elt in arg.elts
                        if isinstance(elt, ast.Constant)
                    )
        elif isinstance(node.value, ast.Constant):
            values[name] = node.value.value
    return (
        values.get("HEADER") if isinstance(values.get("HEADER"), bytes) else None,
        int(values["PKT_LEN"]) if "PKT_LEN" in values else None,
        int(values["TYPE_AA"]) if "TYPE_AA" in values else None,
        int(values["TYPE_BB"]) if "TYPE_BB" in values else None,
    )


@dataclass(frozen=True)
class SerialProtocol:
    header: bytes
    packet_len: int
    emg_type: int
    imu_type: int | None = None
    payload_offset: int = 5

    @classmethod
    def from_collect_or_args(
        cls,
        header_hex: str | None,
        packet_len: int | None,
        emg_type: int | None,
        imu_type: int | None,
        payload_offset: int = 5,
    ) -> "SerialProtocol":
        header, pkt_len, type_aa, type_bb = _load_collect_protocol()
        if header_hex is not None:
            header = bytes.fromhex(header_hex.replace("0x", "").replace(" ", ""))
        if packet_len is not None:
            pkt_len = int(packet_len)
        if emg_type is not None:
            type_aa = int(emg_type)
        if imu_type is not None:
            type_bb = int(imu_type)
        if header is None or pkt_len is None or type_aa is None:
            raise ValueError(
                "Serial protocol is unknown. Restore collect.py or pass "
                "--header-hex, --packet-len, and --emg-type."
            )
        return cls(
            header=header,
            packet_len=pkt_len,
            emg_type=type_aa,
            imu_type=type_bb,
            payload_offset=payload_offset,
        )

    def decode_emg_packet(self, packet: bytes) -> np.ndarray | None:
        if len(packet) != self.packet_len or not packet.startswith(self.header):
            return None
        if packet[len(self.header)] != self.emg_type:
            return None
        payload = packet[self.payload_offset :]
        if len(payload) != 24:
            return None
        values = [
            _int24_be_signed(payload[i], payload[i + 1], payload[i + 2])
            for i in range(0, 24, 3)
        ]
        return np.asarray(values, dtype=np.float32)


class SerialEmgReader(Thread):
    """Background serial reader that emits decoded 8ch EMG samples."""

    def __init__(
        self,
        port: str,
        baud: int,
        protocol: SerialProtocol,
        on_sample: Callable[[float, np.ndarray], None],
        timeout: float = 0.05,
        read_size: int = 256,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baud = int(baud)
        self.timeout = float(timeout)
        self.read_size = int(read_size)
        self.protocol = protocol
        self.on_sample = on_sample
        self._running = Event()
        self._running.set()
        self.samples_read = 0
        self.status = "initializing"

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        import serial

        ser = serial.Serial(self.port, baudrate=self.baud, timeout=self.timeout)
        try:
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            self.status = "connected"
            buf = bytearray()
            while self._running.is_set():
                chunk = ser.read(self.read_size)
                if chunk:
                    buf.extend(chunk)
                if len(buf) > 16 * self.protocol.packet_len:
                    header_idx = buf.find(self.protocol.header)
                    del buf[: max(header_idx, 0)]
                while True:
                    header_idx = buf.find(self.protocol.header)
                    if header_idx < 0:
                        break
                    if header_idx > 0:
                        del buf[:header_idx]
                    if len(buf) < self.protocol.packet_len:
                        break
                    packet = bytes(buf[: self.protocol.packet_len])
                    del buf[: self.protocol.packet_len]
                    emg = self.protocol.decode_emg_packet(packet)
                    if emg is not None:
                        self.samples_read += 1
                        self.on_sample(time.time(), emg)
        finally:
            ser.close()
