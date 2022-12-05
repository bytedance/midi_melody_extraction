# Copyright 2022 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT 
# 
# @Author: Katerina Kosta, Andrew Shaw, Gianluca Micchi
# @Date:   2021-05-21 11:48:11
# @Last Modified by:   Katerina Kosta
# @Last Modified time: 2022-08-29 14:20:10

import os
import heapq
import miditoolkit
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
from collections import defaultdict
import numpy as np
import music21
import logger
from fractions import Fraction
import copy

sqvs_per_beat = 4  # number of semiquavers in a crochet
major_scale_comp = np.array([0, 2, 4, 5, 7, 9, 11])
minor_scale_comp = np.array([0, 2, 3, 5, 7, 8, 10])

pitch_name_classes = {
    "B#": 0,
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


class Note(object):
    """
    A musical note with attributes:

    start    : start time in beats (float)
    duration : duration in beats (float)
    pitch    : pitch number (int)
    track    : channel name (str)
    """

    def __init__(self, start, duration, pitch, track):
        self.start = start
        self.duration = duration
        self.pitch = pitch
        self.track = track

    def __repr__(self):
        return "Note(start={}, duration={}, pitch={}, track={})".format(
            self.start, self.duration, self.pitch, self.track
        )


class KeySignature(object):
    """
    A key signature event with attribute:

    name : key signature (str)
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "KeySignature(name={})".format(self.name)


class TimeSignature(object):
    """
    A time signature event with attribute:

    name : key signature (str)
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "TimeSignature(name={})".format(self.name)


def miditoolkitEvent_to_NoteEvent(note_obj, beat_resol, trackname):
    start_in_beats = note_obj.start / beat_resol
    end_in_beats = note_obj.end / beat_resol
    return Note(
        start=np.round(start_in_beats, 4),
        duration=np.round(end_in_beats - start_in_beats, 4),
        pitch=note_obj.pitch,
        track=trackname,
    )


def NoteEvent_to_miditoolkitEvent(note, beat_resol):
    return ct.Note(
        start=int(note.start * beat_resol),
        end=int((note.start + note.duration) * beat_resol),
        pitch=note.pitch,
        velocity=60,
    )


def read_MIDI_file(filepath):
    midi = miditoolkit.midi.parser.MidiFile(filepath)
    beat_resol = midi.ticks_per_beat

    score = []
    for track in midi.instruments:
        trackname = track.name
        score.append(
            [miditoolkitEvent_to_NoteEvent(note, beat_resol, trackname) for note in track.notes]
        )

    score_flat_parts = [item for sublist in score for item in sublist]
    score_sorted = sorted(score_flat_parts, key=lambda x: x.start)

    return score_sorted


def write_MIDI_melody_file(notes, output_file):
    mido_obj = mid_parser.MidiFile()
    beat_resol = mido_obj.ticks_per_beat
    track = ct.Instrument(program=0, is_drum=False, name="melody")
    mido_obj.instruments = [track]

    for note in notes:
        new_event = NoteEvent_to_miditoolkitEvent(note, beat_resol)
        mido_obj.instruments[0].notes.append(new_event)

    mido_obj.dump(output_file)
    print("Output MIDI file created!", output_file)


def maybe_raise_exception_for_time_signature(filepath):
    folder_path = os.path.dirname(filepath)
    numerator = int(max(np.loadtxt(os.path.join(folder_path, "beat_audio.txt"))[:, 1]))
    return numerator, int(
        max(np.loadtxt(os.path.join(folder_path, "beat_audio.txt"))[:, 1])
    ) not in [1, 2, 4]


def infer_key_sig(fname):
    """Analyse the score with music21 to assign a key and return it (None if unsuccessful)."""
    try:
        m21_score = music21.converter.parse(fname)
        p = music21.analysis.discrete.KrumhanslSchmuckler()

        # Convert music21's notation
        key = (
            p.getSolution(m21_score)
            .name.replace(" major", "M")
            .replace(" minor", "m")
            .replace("-", "b")
        )

        return key
    except:
        logger.error("Impossible to infer the key")


def get_key_signature(filepath):
    """
    If the key-signature metadata file in pop909 contains a single key signature,
    then udpate the score with this one. Otherwise, call music21 key-signature
    estimation algorithm.
    """
    folder_path = os.path.dirname(filepath)
    key_info = np.loadtxt(os.path.join(folder_path, "key_audio.txt"), dtype=np.str)

    if len(key_info.flatten()) == 3:
        _, _, key = key_info
        new_key_string = key.replace(":", "").replace("maj", "M").replace("min", "m")
    else:
        new_key_string = infer_key_sig(filepath)

    return new_key_string


############# PRE-PROCESSING FUNCTION #############


def align_and_quantise_notes(score, distance_threshold=Fraction(1, 4)):
    new_score = copy.deepcopy(score)
    denominators_start = [8, 6]
    denominators_duration = [4]

    def round_nearest_fraction(x, den):
        return Fraction(round(x * den), den)

    def round_grid(x, denominators):
        res = [round_nearest_fraction(x, den) for den in denominators]
        idx = np.argmin([abs(x - r) for r in res])
        return res[idx]

    def find_next_note(notes, start):
        for ne in notes:
            if ne.start > start:
                return ne
        return None

    def quantise_starts(notes, denominators):
        for ne in notes:
            ne.start = round_grid(ne.start, denominators)
        return

    quantise_starts(new_score, denominators_start)
    for i in range(len(new_score)):
        cne = new_score[i]  # current note event
        nne = find_next_note(new_score[i:], cne.start)
        if nne is None:
            cne.duration = round_grid(cne.duration, denominators_duration)
        else:
            if cne.start + cne.duration < nne.start:
                cne.duration = min(
                    nne.start - cne.start, round_grid(cne.duration, denominators_duration)
                )
            elif cne.start + cne.duration == nne.start:
                pass
            elif cne.start + cne.duration - nne.start < distance_threshold:
                cne.duration = nne.start - cne.start
            else:
                cne.duration = round_grid(cne.duration, denominators_duration)
    return new_score


#################### FEATURES ######################


def compute_pitch(notes):
    return np.asarray([e.pitch for e in notes], dtype=np.float32)


def compute_dur(notes):
    return np.asarray([note.duration * sqvs_per_beat for note in notes], dtype=np.float32)


def compute_pos_in_bar(notes, time_signature):
    if time_signature.name == "4/4":
        measure_length = 4
    elif time_signature.name == "3/4":
        measure_length = 3
    else:
        print("pos_in_bar feature currently doesn't support time signatures not in ['3/4', '4/4']")
        return []
    note_starts = [note.start * sqvs_per_beat for note in notes]
    st = [s / sqvs_per_beat for s in note_starts]
    return np.asarray([start % measure_length for start in st], dtype=np.float32)


def compute_in_scale(notes, key_signature):
    pitches = compute_pitch(notes)
    # we assume we only have one key signature.
    pitch_class_number = pitch_name_classes[key_signature.name[:-1]]
    tonality = key_signature.name[-1]
    if tonality == "M":
        scale_degrees = [(pitch_class_number + sd) % 12 for sd in major_scale_comp]
    elif tonality == "m":
        scale_degrees = [(pitch_class_number + sd) % 12 for sd in minor_scale_comp]

    is_note_in_scale = [int(pitch % 12 in scale_degrees) for pitch in pitches]
    return np.asarray(is_note_in_scale, dtype=np.float32)


def compute_overlaps(sorted_items):
    # Computes previous overlaps as we iterate through starts. nlogn time (for sort)
    # assumes sorted items.
    idx2overlap = {}
    minheap = []
    for start, end, idx in sorted_items:
        while len(minheap) > 0 and minheap[0][0] < start:
            heapq.heappop(minheap)
        idx2overlap[idx] = set([i[1] for i in minheap])
        heapq.heappush(minheap, (end, idx))
    return idx2overlap


def compute_simultaneous_notes(notes):
    # compute all previous overlaps
    tups_start = sorted([(n.start, n.duration + n.start, idx) for idx, n in enumerate(notes)])
    forward_overlaps = compute_overlaps(tups_start)

    # compute overlaps on future notes. compute_overlaps is only backwards looking
    tups_endfirst = sorted([(-end, -start, idx) for (start, end, idx) in tups_start])
    backward_overlaps = compute_overlaps(tups_endfirst)

    # compute overlaps does not handle duplicates (aka chords). We must include this
    dup_overlaps = defaultdict(set)
    for start, end, idx in tups_start:
        dup_overlaps[(start, end)].add(idx)

    # check which pitches are active at the start time of a note
    simultaneous_pitches_per_note = []
    for i in range(len(notes)):
        n = notes[i]
        all_overlaps = all_overlaps = (
            forward_overlaps[i]
            | backward_overlaps[i]
            | dup_overlaps[(n.start, n.start + n.duration)]
        )
        active_overlaps = set()
        for idx in all_overlaps:
            overlap = notes[idx]
            if overlap.pitch == []:
                continue  # remove empty pitches
            overlap_end = overlap.start + overlap.duration
            if overlap.start <= notes[i].start < overlap_end:  # only activate notes at start time
                active_overlaps.add(overlap.pitch)
        simultaneous_pitches_per_note.append(active_overlaps)
    return simultaneous_pitches_per_note


def compute_pitch_dist(notes, dist_type):
    # dist_type can be either 'below' or 'above'
    assert dist_type in ["below", "above"]

    # check which pitches are active at the start time of a note
    simultaneous_pitches_per_note = compute_simultaneous_notes(notes)

    pitch_dist = []

    for note_pitch, active_pitches in list(
        zip([note.pitch for note in notes], simultaneous_pitches_per_note)
    ):
        active_pitches = sorted(active_pitches, reverse=dist_type == "above")

        if note_pitch not in active_pitches:  # edge case: note with duration 0
            pitch_dist.append(0)
            continue
        else:
            loc = active_pitches.index(note_pitch)
            if len(active_pitches) == 1 or loc == 0:  # case where the note is already the lowest
                pitch_dist.append(0)
                continue
            pitch_dist.append(abs(active_pitches[loc - 1] - note_pitch))
    return np.asarray(pitch_dist, dtype=np.float32)


def compute_is_melody_note(notes):
    is_melody_note = [int(note.track == "MELODY") for note in notes]
    return np.asarray(is_melody_note, dtype=np.float32)
