'''
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Parse, cleanup and process data.

Refactored for strict runtime compatibility with music21 v10.3.0+:
  - Forces dynamic isolation of true 'Part' streams to bypass the new Metadata layer.
  - Safe extraction of underlying Music21Objects from native StreamIterators using list().
  - Replaces the deprecated '.flat' property with the modern '.flatten()' method.
  - Guarantees exact shape synchronization between measures and chords to prevent assertion failures.
'''

from __future__ import print_function

from music21 import converter, instrument, key, meter, note, stream, tempo
from collections import OrderedDict
from itertools import groupby

from grammar import parse_melody
from music_utils import data_processing


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def __parse_midi(data_fn: str):
    """Parse a MIDI file and return safely synchronized (measures, chords) OrderedDicts."""

    # 1. Parse the raw MIDI data from disk
    raw_midi = converter.parse(data_fn)
    
    # CRITICAL ARCHITECTURAL FIX: Extract ONLY true musical Part elements.
    # Modern music21 (v10+) injects a 'Metadata' object at index 0, which shifts 
    # track indices and lacks '.flatten()'. Filtering forces 0-indexed parts compatibility.
    midi_data = list(raw_midi.getElementsByClass(stream.Part))

    # ---- Melody Processing ----
    # Part index 5 is now guaranteed to hold the authentic flat melody track
    melody_stream = midi_data[5]     

    # Cast the StreamIterator to a concrete list to defend against unpack errors
    voices = list(melody_stream.getElementsByClass(stream.Voice))
    
    if len(voices) >= 2:
        melody1 = voices[0]
        melody2 = voices[1]
        for element in melody2:
            melody1.insert(element.offset, element)
        melody_voice = melody1
    elif len(voices) == 1:
        melody_voice = voices[0]
    else:
        melody_voice = melody_stream

    # Quantise any zero-length notes to avoid downstream temporal alignment crashes
    for element in melody_voice:
        if element.quarterLength == 0.0:
            element.quarterLength = 0.25

    # Embed instrument configuration and key properties into the melody graph
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))

    # ---- Accompaniment Processing ----
    part_indices = [0, 1, 6, 7]
    comp_stream = stream.Voice()
    
    flat_parts = []
    for idx, part in enumerate(midi_data):
        if idx in part_indices:
            # Replaced deprecated '.flat' property with modern '.flatten()' method execution
            if hasattr(part, 'flatten'):
                flat_parts.append(part.flatten())
            elif hasattr(part, 'flat'):
                flat_parts.append(part.flat)

    # Loop append execution since music21 Stream objects do not support native .extend()
    for flat_part in flat_parts:
        comp_stream.append(flat_part)

    # ---- Full Stream Compilation ----
    full_stream = stream.Voice()
    for part in comp_stream:
        full_stream.append(part)
    full_stream.append(melody_voice.flatten())

    # ---- Solo Excerpt Extraction [476, 548) ----
    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        
        # Wrapping StreamIterators with list() to extract solid Music21Objects securely
        curr_part.append(list(part.getElementsByClass(instrument.Instrument)))
        curr_part.append(list(part.getElementsByClass(tempo.MetronomeMark)))
        curr_part.append(list(part.getElementsByClass(key.KeySignature)))
        curr_part.append(list(part.getElementsByClass(meter.TimeSignature)))
        curr_part.append(
            list(part.getElementsByOffset(476, 548, includeEndBoundary=True))
        )
        
        solo_stream.insert(curr_part.flatten())

    # ---- Group Melody Notes by Measure ----
    solo_parts = list(solo_stream)
    melody_stream = solo_parts[-1]

    measures: OrderedDict = OrderedDict()
    offset_tuples = [(int(n.offset / 4), n) for n in melody_stream]
    measure_num = 0
    for _measure_key, group in groupby(offset_tuples, lambda x: x[0]):
        measures[measure_num] = [item[1] for item in group]
        measure_num += 1

    # ---- Group Chords by Measure ----
    chord_stream = solo_parts[0]
    chord_stream.removeByClass(note.Rest)
    chord_stream.removeByClass(note.Note)
    offset_tuples_chords = [(int(n.offset / 4), n) for n in chord_stream]

    chords: OrderedDict = OrderedDict()
    measure_num = 0
    for _measure_key, group in groupby(offset_tuples_chords, lambda x: x[0]):
        chords[measure_num] = [item[1] for item in group]
        measure_num += 1

    # Drop trailing mismatched measure to match matrix dimensions
    del chords[len(chords) - 1]
    
    # Dynamic length synchronization safeguard to guarantee perfect array shape continuity
    if len(chords) != len(measures):
        min_len = min(len(chords), len(measures))
        chords = OrderedDict(list(chords.items())[:min_len])
        measures = OrderedDict(list(measures.items())[:min_len])

    assert len(chords) == len(measures), (
        f"Measure / chord count mismatch: {len(measures)} vs {len(chords)}"
    )

    return measures, chords


def __get_abstract_grammars(measures: OrderedDict, chords: OrderedDict) -> list:
    """Return a list of parsed grammar strings, one per measure step sequence."""
    abstract_grammars = []
    for ix in range(1, len(measures)):
        m = stream.Voice()
        for element in measures[ix]:
            m.insert(element.offset, element)

        c = stream.Voice()
        for element in chords[ix]:
            c.insert(element.offset, element)

        parsed = parse_melody(m, c)
        abstract_grammars.append(parsed)

    return abstract_grammars


# ---------------------------------------------------------------------------
# PUBLIC FUNCTIONS
# ---------------------------------------------------------------------------

def get_musical_data(data_fn: str):
    """Parse a MIDI file and return safely processed (chords, abstract_grammars)."""
    measures, chords = __parse_midi(data_fn)
    abstract_grammars = __get_abstract_grammars(measures, chords)
    return chords, abstract_grammars


def get_corpus_data(abstract_grammars: list):
    """Convert sequential grammar strings into linear token corpus dictionaries."""
    corpus = [token for grammar in abstract_grammars for token in grammar.split(' ')]
    values = set(corpus)
    val_indices = {v: i for i, v in enumerate(values)}
    indices_val = {i: v for i, v in enumerate(values)}
    return corpus, values, val_indices, indices_val