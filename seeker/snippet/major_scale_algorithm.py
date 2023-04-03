#date: 2023-04-03T17:10:13Z
#url: https://api.github.com/gists/5840f929b604006309c7ca6c88708d3b
#owner: https://api.github.com/users/yulleyi

def generate_major_scale(root_note):
    major_scale_pattern = [2, 2, 1, 2, 2, 2, 1]
    chromatic_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    root_note_index = chromatic_notes.index(root_note)
    major_scale_notes = []

    for step in major_scale_pattern:
        root_note_index = (root_note_index + step) % 12
        major_scale_notes.append(chromatic_notes[root_note_index])

    return major_scale_notes