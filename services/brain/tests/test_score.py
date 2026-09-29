from __future__ import annotations

import importlib
import importlib.util
import struct
from pathlib import Path

import pytest


PPQN = 24


def _vlq(value: int) -> bytes:
    encoded = [value & 0x7F]
    value >>= 7
    while value:
        encoded.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(encoded)


def _meta(delta: int, kind: int, payload: bytes) -> bytes:
    return _vlq(delta) + bytes((0xFF, kind)) + _vlq(len(payload)) + payload


def _channel(delta: int, status: int, *data: int) -> bytes:
    return _vlq(delta) + bytes((status, *data))


def _track(*events: bytes) -> bytes:
    body = b"".join(events) + _meta(0, 0x2F, b"")
    return b"MTrk" + struct.pack(">I", len(body)) + body


def _midi_bytes() -> bytes:
    conductor = _track(
        _meta(0, 0x03, b"Conductor"),
        _meta(0, 0x58, bytes((4, 2, 24, 8))),
        _meta(0, 0x51, (500_000).to_bytes(3, "big")),
        _meta(96, 0x51, (1_000_000).to_bytes(3, "big")),
    )
    drums = _track(
        _meta(0, 0x03, b"Drums"),
        _channel(0, 0x99, 36, 64),
        _channel(0, 0x99, 38, 127),
        _channel(24, 0x89, 36, 0),
        _channel(0, 0x89, 38, 0),
        _channel(24, 0x99, 42, 80),
        _channel(48, 0x99, 46, 96),
        _channel(96, 0x99, 88, 55),
        _channel(96, 0x99, 47, 100),
    )
    return (
        b"MThd"
        + struct.pack(">IHHH", 6, 1, 2, PPQN)
        + conductor
        + drums
    )


def _variable_meter_midi_bytes(
    *, meter_change_tick: int = 384, include_later_change: bool = True
) -> bytes:
    conductor_events = [
        _meta(0, 0x03, b"Conductor"),
        _meta(0, 0x58, bytes((4, 2, 24, 8))),
        _meta(0, 0x51, (500_000).to_bytes(3, "big")),
    ]
    previous_tick = 0
    for tick, numerator, denominator_power in (
        [(meter_change_tick, 3, 2)]
        + ([(600, 2, 2)] if include_later_change else [])
    ):
        conductor_events.append(
            _meta(
                tick - previous_tick,
                0x58,
                bytes((numerator, denominator_power, 24, 8)),
            )
        )
        previous_tick = tick

    drum_events = [_meta(0, 0x03, b"Drums")]
    previous_tick = 0
    for tick in (0, 383, 384, 455, 599, 600):
        drum_events.append(_channel(tick - previous_tick, 0x99, 36, 100))
        previous_tick = tick

    return (
        b"MThd"
        + struct.pack(">IHHH", 6, 1, 2, PPQN)
        + _track(*conductor_events)
        + _track(*drum_events)
    )


def _importer():
    if importlib.util.find_spec("flydrummer_brain.score") is None:
        return None
    return importlib.import_module("flydrummer_brain.score").import_midi


def test_importer_exposes_normalized_score_api(tmp_path: Path) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "fixture.mid"
    source.write_bytes(_midi_bytes())
    score = importer(source, start_bar=1, bars=3)
    assert score.midi_track_count == 2
    assert score.ticks_per_quarter == PPQN
    assert score.start_tick == 0
    assert score.end_tick == 288


def test_tempo_changes_are_integrated_into_absolute_song_time(tmp_path: Path) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "fixture.mid"
    source.write_bytes(_midi_bytes())
    score = importer(source, start_bar=1, bars=3)

    assert [(change.tick, change.time_seconds, change.microseconds_per_quarter) for change in score.tempo_map] == [
        (0, 0.0, 500_000),
        (96, 2.0, 1_000_000),
    ]
    assert next(event.time_seconds for event in score.events if event.source_tick == 96) == 2.0


def test_simultaneous_notes_and_velocity_are_preserved(tmp_path: Path) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "fixture.mid"
    source.write_bytes(_midi_bytes())
    score = importer(source, start_bar=1, bars=3)
    first_tick = [event for event in score.events if event.source_tick == 0]

    assert [event.instrument for event in first_tick] == ["kick", "snare"]
    assert [event.velocity for event in first_tick] == pytest.approx([64 / 127, 1.0])


def test_unknown_note_is_reported_without_becoming_a_drum_event(tmp_path: Path) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "fixture.mid"
    source.write_bytes(_midi_bytes())
    score = importer(source, start_bar=1, bars=3)

    assert [(item.note, item.source_tick, item.velocity) for item in score.unknown_notes] == [(88, 192, 55)]
    assert all(event.source_tick != 192 for event in score.events)


def test_time_range_is_start_inclusive_and_end_exclusive(tmp_path: Path) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "fixture.mid"
    source.write_bytes(_midi_bytes())
    score = importer(source, start_bar=2, bars=1)

    assert score.start_tick == 96
    assert score.end_tick == 192
    assert [(event.source_tick, event.instrument) for event in score.events] == [(96, "open_hat")]
    assert [change.tick for change in score.tempo_map] == [96]


def test_future_time_signature_change_does_not_block_earlier_bars(tmp_path: Path) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "variable-meter.mid"
    source.write_bytes(_variable_meter_midi_bytes())

    score = importer(source, start_bar=1, bars=4)

    assert (score.start_tick, score.end_tick) == (0, 384)
    assert (score.time_signature_numerator, score.time_signature_denominator) == (4, 4)
    assert [event.source_tick for event in score.events] == [0, 383]


def test_bar_ranges_follow_meter_changes_at_bar_boundaries(tmp_path: Path) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "variable-meter.mid"
    source.write_bytes(_variable_meter_midi_bytes())

    score = importer(source, start_bar=5, bars=3)

    assert (score.start_tick, score.end_tick) == (384, 600)
    assert (score.time_signature_numerator, score.time_signature_denominator) == (3, 4)
    assert [event.source_tick for event in score.events] == [384, 455, 599]


def test_meter_change_inside_a_bar_is_rejected_as_ambiguous(tmp_path: Path) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "ambiguous-meter.mid"
    source.write_bytes(_variable_meter_midi_bytes(meter_change_tick=385, include_later_change=False))

    with pytest.raises(importer.__globals__["MidiImportError"], match="(?i)bar boundary"):
        importer(source, start_bar=1, bars=4)


def test_truncated_midi_is_reported_as_import_error(tmp_path: Path) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "broken.mid"
    source.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x18MTrk\x00\x00\x00\x20\x00\xff")

    with pytest.raises(importer.__globals__["MidiImportError"], match="(?i)truncated|поврежд|обрезан"):
        importer(source)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"not a MIDI file", "header"),
        (b"MThd" + struct.pack(">IHHH", 6, 0, 1, 0), "division"),
        (b"MThd" + struct.pack(">IHHH", 6, 2, 1, 480), "Unsupported Standard MIDI format"),
        (
            b"MThd"
            + struct.pack(">IHHH", 6, 0, 1, 480)
            + b"MTrk"
            + struct.pack(">I", 5)
            + b"\x81\x80\x80\x80\x00",
            "variable-length value",
        ),
    ],
    ids=("missing-header", "zero-division", "unsupported-format", "malformed-vlq"),
)
def test_invalid_midi_headers_and_events_raise_import_error(
    tmp_path: Path, contents: bytes, message: str
) -> None:
    importer = _importer()
    assert importer is not None, "flydrummer_brain.score.import_midi is not implemented"

    source = tmp_path / "invalid.mid"
    source.write_bytes(contents)

    with pytest.raises(importer.__globals__["MidiImportError"], match=message):
        importer(source)
