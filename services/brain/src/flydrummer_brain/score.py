"""Import General MIDI drum events from a Standard MIDI File."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


class MidiImportError(ValueError):
    """Raised when a MIDI file is malformed or cannot be mapped safely."""


DRUM_NOTE_MAP: dict[int, str] = {
    35: "kick",
    36: "kick",
    38: "snare",
    40: "snare",
    41: "tom",
    42: "closed_hat",
    43: "tom",
    44: "closed_hat",
    45: "tom",
    46: "open_hat",
    47: "tom",
    48: "tom",
    49: "crash",
    50: "tom",
    51: "ride",
    52: "crash",
    53: "ride",
    55: "crash",
    57: "crash",
    59: "ride",
}


@dataclass(frozen=True)
class DrumEvent:
    time_seconds: float
    instrument: str
    velocity: float
    source_tick: int
    note: int
    track_index: int
    track_name: str
    channel: int


@dataclass(frozen=True)
class TempoChange:
    tick: int
    time_seconds: float
    microseconds_per_quarter: int
    bpm: float


@dataclass(frozen=True)
class UnknownNote:
    source_tick: int
    note: int
    velocity: int
    track_index: int
    track_name: str
    channel: int


@dataclass(frozen=True)
class SourceTrack:
    index: int
    name: str
    channels: tuple[int, ...]
    note_on_count: int


@dataclass(frozen=True)
class ScoreFragment:
    source_filename: str
    source_sha256: str
    source_size_bytes: int
    midi_format: int
    midi_track_count: int
    ticks_per_quarter: int
    start_bar: int
    bars: int
    start_tick: int
    end_tick: int
    start_time_seconds: float
    end_time_seconds: float
    time_signature_numerator: int
    time_signature_denominator: int
    source_tracks: tuple[SourceTrack, ...]
    events: tuple[DrumEvent, ...]
    tempo_map: tuple[TempoChange, ...]
    unknown_notes: tuple[UnknownNote, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable score fragment and its mapping evidence."""
        event_counts = Counter(event.instrument for event in self.events)
        unknown_counts = Counter(note.note for note in self.unknown_notes)
        return {
            "schema_version": 1,
            "source": {
                "filename": self.source_filename,
                "sha256": self.source_sha256,
                "size_bytes": self.source_size_bytes,
            },
            "midi_format": self.midi_format,
            "midi_track_count": self.midi_track_count,
            "ticks_per_quarter": self.ticks_per_quarter,
            "fragment": {
                "start_bar": self.start_bar,
                "bars": self.bars,
                "start_tick": self.start_tick,
                "end_tick_exclusive": self.end_tick,
                "start_time_seconds": self.start_time_seconds,
                "end_time_seconds": self.end_time_seconds,
                "time_signature": [
                    self.time_signature_numerator,
                    self.time_signature_denominator,
                ],
            },
            "source_tracks": [asdict(track) for track in self.source_tracks],
            "tempo_map": [asdict(change) for change in self.tempo_map],
            "events": [asdict(event) for event in self.events],
            "mapping_report": {
                "mapped_event_counts": dict(sorted(event_counts.items())),
                "unmapped_note_counts": {
                    str(note): count for note, count in sorted(unknown_counts.items())
                },
                "unmapped_notes": [asdict(note) for note in self.unknown_notes],
                "gm_note_mapping": {
                    str(note): instrument for note, instrument in sorted(DRUM_NOTE_MAP.items())
                },
            },
        }


@dataclass(frozen=True)
class _RawNote:
    tick: int
    note: int
    velocity: int
    track_index: int
    track_name: str
    channel: int
    order: int


@dataclass(frozen=True)
class _RawTrack:
    index: int
    name: str
    notes: tuple[_RawNote, ...]
    tempos: tuple[tuple[int, int], ...]
    time_signatures: tuple[tuple[int, int, int], ...]
    channels: tuple[int, ...]


def _read_vlq(data: bytes, position: int, limit: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if position >= limit:
            raise MidiImportError("Truncated MIDI variable-length value")
        byte = data[position]
        position += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, position
    raise MidiImportError("Invalid MIDI variable-length value exceeds four bytes")


def _parse_track(data: bytes, start: int, end: int, track_index: int) -> _RawTrack:
    position = start
    tick = 0
    running_status: int | None = None
    track_name = ""
    notes: list[_RawNote] = []
    tempos: list[tuple[int, int]] = []
    time_signatures: list[tuple[int, int, int]] = []
    channels: set[int] = set()
    order = 0
    found_end = False

    while position < end:
        delta, position = _read_vlq(data, position, end)
        tick += delta
        if position >= end:
            raise MidiImportError(f"Truncated event in MIDI track {track_index}")

        first = data[position]
        if first & 0x80:
            status = first
            position += 1
            if 0x80 <= status <= 0xEF:
                running_status = status
            elif status not in (0xF0, 0xF7, 0xFF):
                raise MidiImportError(f"Unsupported MIDI status 0x{status:02x}")
        else:
            if running_status is None:
                raise MidiImportError(f"Missing running status in MIDI track {track_index}")
            status = running_status

        if status == 0xFF:
            if position >= end:
                raise MidiImportError(f"Truncated meta event in MIDI track {track_index}")
            meta_type = data[position]
            position += 1
            length, position = _read_vlq(data, position, end)
            payload_end = position + length
            if payload_end > end:
                raise MidiImportError(f"Truncated meta payload in MIDI track {track_index}")
            payload = data[position:payload_end]
            position = payload_end
            if meta_type == 0x03:
                track_name = payload.decode("utf-8", errors="replace").strip()
            elif meta_type == 0x51:
                if len(payload) != 3:
                    raise MidiImportError(f"Invalid tempo meta event in MIDI track {track_index}")
                tempo = int.from_bytes(payload, "big")
                if tempo == 0:
                    raise MidiImportError("MIDI tempo must be greater than zero")
                tempos.append((tick, tempo))
            elif meta_type == 0x58:
                if len(payload) < 2 or payload[1] > 7:
                    raise MidiImportError(f"Invalid time signature in MIDI track {track_index}")
                time_signatures.append((tick, payload[0], 1 << payload[1]))
            elif meta_type == 0x2F:
                if length != 0:
                    raise MidiImportError(f"Invalid end-of-track event in MIDI track {track_index}")
                found_end = True
                if position != end:
                    raise MidiImportError(f"Data follows end-of-track in MIDI track {track_index}")
                break
        elif status in (0xF0, 0xF7):
            length, position = _read_vlq(data, position, end)
            position += length
            if position > end:
                raise MidiImportError(f"Truncated system-exclusive event in MIDI track {track_index}")
        elif 0x80 <= status <= 0xEF:
            message_kind = status >> 4
            channel = status & 0x0F
            data_length = 1 if message_kind in (0xC, 0xD) else 2
            if position + data_length > end:
                raise MidiImportError(f"Truncated channel event in MIDI track {track_index}")
            values = data[position : position + data_length]
            if any(value & 0x80 for value in values):
                raise MidiImportError(f"Invalid channel data byte in MIDI track {track_index}")
            position += data_length
            channels.add(channel)
            if message_kind == 0x9 and values[1] > 0:
                notes.append(
                    _RawNote(
                        tick=tick,
                        note=values[0],
                        velocity=values[1],
                        track_index=track_index,
                        track_name=track_name,
                        channel=channel,
                        order=order,
                    )
                )
                order += 1
        else:
            raise MidiImportError(f"Unsupported MIDI event status 0x{status:02x}")

    if not found_end:
        raise MidiImportError(f"Missing end-of-track event in MIDI track {track_index}")

    # Track names can occur after note messages, so refresh the retained name.
    notes = [
        _RawNote(
            tick=note.tick,
            note=note.note,
            velocity=note.velocity,
            track_index=note.track_index,
            track_name=track_name,
            channel=note.channel,
            order=note.order,
        )
        for note in notes
    ]
    return _RawTrack(
        index=track_index,
        name=track_name,
        notes=tuple(notes),
        tempos=tuple(tempos),
        time_signatures=tuple(time_signatures),
        channels=tuple(sorted(channels)),
    )


def _parse_midi(data: bytes) -> tuple[int, int, tuple[_RawTrack, ...]]:
    if len(data) < 14 or data[:4] != b"MThd":
        raise MidiImportError("File does not contain a complete Standard MIDI header")
    header_length = int.from_bytes(data[4:8], "big")
    if header_length < 6 or 8 + header_length > len(data):
        raise MidiImportError("Invalid or truncated Standard MIDI header")
    midi_format = int.from_bytes(data[8:10], "big")
    track_count = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if midi_format not in (0, 1):
        raise MidiImportError(f"Unsupported Standard MIDI format {midi_format}; expected format 0 or 1")
    if track_count == 0 or (midi_format == 0 and track_count != 1):
        raise MidiImportError("Invalid track count for Standard MIDI format")
    if division & 0x8000 or division == 0:
        raise MidiImportError("SMPTE or zero MIDI time division is not supported; expected PPQN")

    position = 8 + header_length
    tracks: list[_RawTrack] = []
    for track_index in range(track_count):
        if position + 8 > len(data) or data[position : position + 4] != b"MTrk":
            raise MidiImportError(f"Missing or truncated MIDI track chunk {track_index}")
        chunk_length = int.from_bytes(data[position + 4 : position + 8], "big")
        chunk_start = position + 8
        chunk_end = chunk_start + chunk_length
        if chunk_end > len(data):
            raise MidiImportError(f"Truncated MIDI track chunk {track_index}")
        tracks.append(_parse_track(data, chunk_start, chunk_end, track_index))
        position = chunk_end
    if position != len(data):
        raise MidiImportError("Unexpected bytes after declared MIDI tracks")
    return midi_format, division, tuple(tracks)


def _deduplicate_events(
    values: list[tuple[int, int]], label: str
) -> list[tuple[int, int]]:
    by_tick: dict[int, set[int]] = {}
    for tick, value in values:
        by_tick.setdefault(tick, set()).add(value)
    conflicts = [tick for tick, at_tick in by_tick.items() if len(at_tick) != 1]
    if conflicts:
        raise MidiImportError(f"Conflicting {label} events at tick(s): {sorted(conflicts)}")
    return [(tick, next(iter(at_tick))) for tick, at_tick in sorted(by_tick.items())]


def _seconds_at(tick: int, ppqn: int, tempos: list[tuple[int, int]]) -> float:
    elapsed = 0.0
    prior_tick = 0
    current_tempo = 500_000
    for change_tick, tempo in tempos:
        if change_tick > tick:
            break
        if change_tick > prior_tick:
            elapsed += (change_tick - prior_tick) * current_tempo / (ppqn * 1_000_000)
            prior_tick = change_tick
        current_tempo = tempo
    if tick > prior_tick:
        elapsed += (tick - prior_tick) * current_tempo / (ppqn * 1_000_000)
    return elapsed


def _ticks_per_bar(ppqn: int, numerator: int, denominator: int) -> int:
    ticks_numerator = ppqn * numerator * 4
    if ticks_numerator % denominator:
        raise MidiImportError("Time signature does not align to an integer MIDI tick bar")
    return ticks_numerator // denominator


def _bar_segments(
    meter_changes: list[tuple[int, int, int]], ppqn: int
) -> list[tuple[int, int, int, int]]:
    """Return (first bar, start tick, numerator, denominator) for each meter."""
    if not meter_changes or meter_changes[0][0] > 0:
        meter_changes = [(0, 4, 4), *meter_changes]

    first_tick, numerator, denominator = meter_changes[0]
    if first_tick != 0:
        raise MidiImportError("Time-signature map must begin at tick zero")
    segments = [(1, 0, numerator, denominator)]
    current_bar = 1
    current_tick = 0
    current_meter = (numerator, denominator)

    for tick, numerator, denominator in meter_changes[1:]:
        next_meter = (numerator, denominator)
        if next_meter == current_meter:
            continue
        ticks_per_bar = _ticks_per_bar(ppqn, *current_meter)
        elapsed_ticks = tick - current_tick
        if elapsed_ticks % ticks_per_bar:
            raise MidiImportError(
                f"Time-signature change at tick {tick} is not on a bar boundary"
            )
        current_bar += elapsed_ticks // ticks_per_bar
        current_tick = tick
        current_meter = next_meter
        segments.append((current_bar, current_tick, numerator, denominator))

    return segments


def _tick_at_bar(
    bar_number: int, ppqn: int, segments: list[tuple[int, int, int, int]]
) -> int:
    first_bar, start_tick, numerator, denominator = segments[0]
    for segment in segments[1:]:
        if segment[0] > bar_number:
            break
        first_bar, start_tick, numerator, denominator = segment
    return start_tick + (bar_number - first_bar) * _ticks_per_bar(
        ppqn, numerator, denominator
    )


def import_midi(path: str | Path, *, start_bar: int = 1, bars: int = 16) -> ScoreFragment:
    """Import a bar-bounded fragment from a format 0/1 MIDI with a named drum track.

    Bar numbering is one-based. Events use absolute song seconds, and the selected
    tick range is start-inclusive and end-exclusive. Drum lanes follow the
    General MIDI percussion note map in ``DRUM_NOTE_MAP``; unrecognized notes are
    returned separately so they cannot disappear without a report.
    """
    if not isinstance(start_bar, int) or not isinstance(bars, int):
        raise ValueError("start_bar and bars must be integers")
    if start_bar < 1 or bars < 1:
        raise ValueError("start_bar and bars must be positive integers")

    source_path = Path(path)
    try:
        source_data = source_path.read_bytes()
    except OSError as exc:
        raise MidiImportError(f"Cannot read MIDI input {source_path}: {exc}") from exc
    midi_format, ppqn, tracks = _parse_midi(source_data)

    named_drum_tracks = [track for track in tracks if "drum" in track.name.casefold()]
    if named_drum_tracks:
        selected_tracks = named_drum_tracks
    else:
        selected_tracks = [track for track in tracks if 9 in track.channels]
    if not selected_tracks:
        raise MidiImportError("No drum track identified by track name or MIDI percussion channel 10")

    raw_time_signatures = [
        event
        for track in tracks
        for event in track.time_signatures
    ]
    unique_meters: dict[int, set[tuple[int, int]]] = {}
    for tick, numerator, denominator in raw_time_signatures:
        if numerator == 0:
            raise MidiImportError("MIDI time-signature numerator must be greater than zero")
        unique_meters.setdefault(tick, set()).add((numerator, denominator))
    conflicting_meter_ticks = [tick for tick, values in unique_meters.items() if len(values) > 1]
    if conflicting_meter_ticks:
        raise MidiImportError(f"Conflicting time signatures at tick(s): {sorted(conflicting_meter_ticks)}")
    meter_changes = sorted(
        (tick, *next(iter(values))) for tick, values in unique_meters.items()
    )
    bar_segments = _bar_segments(meter_changes, ppqn)
    start_tick = _tick_at_bar(start_bar, ppqn, bar_segments)
    end_tick = _tick_at_bar(start_bar + bars, ppqn, bar_segments)
    _, _, numerator, denominator = next(
        segment
        for segment in reversed(bar_segments)
        if segment[0] <= start_bar
    )

    raw_tempos = [event for track in tracks for event in track.tempos]
    tempos = _deduplicate_events(raw_tempos, "tempo")
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0, 500_000))

    selected_notes = sorted(
        (
            note
            for track in selected_tracks
            for note in track.notes
            if start_tick <= note.tick < end_tick
        ),
        key=lambda note: (note.tick, note.track_index, note.order),
    )
    events: list[DrumEvent] = []
    unknown_notes: list[UnknownNote] = []
    for note in selected_notes:
        instrument = DRUM_NOTE_MAP.get(note.note)
        if instrument is None:
            unknown_notes.append(
                UnknownNote(
                    source_tick=note.tick,
                    note=note.note,
                    velocity=note.velocity,
                    track_index=note.track_index,
                    track_name=note.track_name,
                    channel=note.channel,
                )
            )
            continue
        events.append(
            DrumEvent(
                time_seconds=_seconds_at(note.tick, ppqn, tempos),
                instrument=instrument,
                velocity=note.velocity / 127,
                source_tick=note.tick,
                note=note.note,
                track_index=note.track_index,
                track_name=note.track_name,
                channel=note.channel,
            )
        )

    relevant_tempos = [
        event for event in tempos if start_tick <= event[0] < end_tick
    ]
    if not relevant_tempos or relevant_tempos[0][0] > start_tick:
        prior_tempos = [event for event in tempos if event[0] <= start_tick]
        active_tempo = prior_tempos[-1][1] if prior_tempos else 500_000
        relevant_tempos.insert(0, (start_tick, active_tempo))
    tempo_map = tuple(
        TempoChange(
            tick=tick,
            time_seconds=_seconds_at(tick, ppqn, tempos),
            microseconds_per_quarter=tempo,
            bpm=60_000_000 / tempo,
        )
        for tick, tempo in relevant_tempos
    )

    source_tracks = tuple(
        SourceTrack(
            index=track.index,
            name=track.name,
            channels=track.channels,
            note_on_count=sum(start_tick <= note.tick < end_tick for note in track.notes),
        )
        for track in selected_tracks
    )
    return ScoreFragment(
        source_filename=source_path.name,
        source_sha256=sha256(source_data).hexdigest(),
        source_size_bytes=len(source_data),
        midi_format=midi_format,
        midi_track_count=len(tracks),
        ticks_per_quarter=ppqn,
        start_bar=start_bar,
        bars=bars,
        start_tick=start_tick,
        end_tick=end_tick,
        start_time_seconds=_seconds_at(start_tick, ppqn, tempos),
        end_time_seconds=_seconds_at(end_tick, ppqn, tempos),
        time_signature_numerator=numerator,
        time_signature_denominator=denominator,
        source_tracks=source_tracks,
        events=tuple(events),
        tempo_map=tempo_map,
        unknown_notes=tuple(unknown_notes),
    )
