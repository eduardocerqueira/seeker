#date: 2023-09-12T17:06:40Z
#url: https://api.github.com/gists/51470a3a901c1cbfe4b464dc5f9c8a4a
#owner: https://api.github.com/users/aniciya777

from __future__ import annotations
from typing import List, Optional

N = 7
PITCHES = ["до", "ре", "ми", "фа", "соль", "ля", "си"]
LONG_PITCHES = ["до-о", "ре-э", "ми-и", "фа-а", "со-оль", "ля-а", "си-и"]
INTERVALS = ["прима", "секунда", "терция", "кварта", "квинта", "секста", "септима"]


class Note:
    NOTES = {
        'до': 'до-о',
        'ре': 'ре-э',
        'ми': 'ми-и',
        'фа': 'фа-а',
        'соль': 'со-оль',
        'ля': 'ля-а',
        'си': 'си-и',
    }

    @property
    def _index(self) -> int:
        return PITCHES.index(self._pitch)

    def __gt__(self, other: Note) -> bool:
        return self._index > other._index

    def __ge__(self, other: Note) -> bool:
        return self._index >= other._index

    def __lt__(self, other: Note) -> bool:
        return self._index < other._index

    def __le__(self, other: Note) -> bool:
        return self._index <= other._index

    def __eq__(self, other: Note) -> bool:
        return self._pitch == other._pitch

    def __ne__(self, other: Note) -> bool:
        return self._pitch != other._pitch

    def __init__(self, pitch: str, is_long: bool = False):
        self._pitch = pitch
        self._duration = is_long

    def play(self) -> None:
        print(self)

    def __str__(self) -> str:
        if self._duration:
            return self.NOTES[self._pitch]
        return self._pitch

    def __lshift__(self, other: int) -> Note:
        pitch = PITCHES[(self._index - other) % 7]
        return Note(pitch, self._duration)

    def __rshift__(self, other: int) -> Note:
        return self << -other

    def get_interval(self, other: Note) -> str:
        return INTERVALS[abs(self._index - other._index)]


class LoudNote(Note):
    def __str__(self) -> str:
        return super().__str__().upper()


class DefaultNote(Note):
    def __init__(self, pitch: str = 'до', is_long: bool = False):
        super().__init__(pitch, is_long)


class NoteWithOctave(Note):
    def __init__(self, pitch: str, octave: str, is_long: bool = False):
        super().__init__(pitch, is_long)
        self._octave = octave

    def __str__(self) -> str:
        return f'{super().__str__()} ({self._octave})'


class Melody:
    def __init__(self, notes: Optional[List[Note]] = None):
        self._notes: List[Note] = notes or []

    def __str__(self) -> str:
        return ', '.join(map(str, self._notes)).capitalize()

    def append(self, note: Note) -> None:
        self._notes.append(note)

    def remove_last(self) -> None:
        self._notes.pop()

    def replace_last(self, note: Note) -> None:
        self._notes[-1] = note

    def clear(self) -> None:
        self._notes.clear()

    def __len__(self) -> int:
        return len(self._notes)

    def __rshift__(self, other: int) -> Melody:
        if other < 0:
            return self << -other
        max_index = 0
        if self._notes:
            max_index = max(note._index for note in self._notes)
        if max_index + other >= 7:
            return self.__copy__()
        return Melody([note >> other for note in self._notes])

    def __lshift__(self, other: int) -> Melody:
        if other < 0:
            return self >> -other
        min_index = 6
        if self._notes:
            min_index = min(note._index for note in self._notes)
        if min_index - other < 0:
            return self.__copy__()
        return Melody([note << other for note in self._notes])

    def __copy__(self) -> Melody:
        new_melody = Melody()
        new_melody._notes = self._notes.copy()
        return new_melody

