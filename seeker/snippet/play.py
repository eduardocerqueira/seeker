#date: 2025-09-11T17:08:38Z
#url: https://api.github.com/gists/fdfcbb16004dbd219200a644043902f3
#owner: https://api.github.com/users/MultiMote

import serial
import mido
import time

s = serial.Serial(
    port="COM27",
    baudrate=115200,
)


def motor(speed=0):
    cmd = f"test motor {int(speed)} 100"
    if speed == 0:
        # motor off
        cmd = "test motor "
    s.write(cmd.encode("ascii"))


def note_to_frequency(note):
    """Convert MIDI note number to frequency in Hz"""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def play_midi_track(midi_file_path):
    """Play the first track of a MIDI file with proper timing"""
    try:
        # Load MIDI file
        midi_file = mido.MidiFile(midi_file_path)

        # Get first track
        if not midi_file.tracks:
            print("No tracks found in MIDI file")
            return

        first_track = midi_file.tracks[0]
        print(f"Playing first track with {len(first_track)} events")

        # Convert MIDI ticks to seconds
        ticks_per_beat = midi_file.ticks_per_beat
        current_tempo = 500000  # Default tempo (500000 microseconds per beat = 120 BPM)

        def ticks_to_seconds(ticks, tempo):
            """Convert MIDI ticks to seconds using current tempo"""
            return mido.tick2second(ticks, ticks_per_beat, tempo)

        # Process MIDI events with proper timing
        current_time = 0
        start_time = time.time()
        active_notes = {}  # Dictionary to track active notes and their start times

        for msg in first_track:
            # Convert delta time to seconds and wait
            delta_seconds = ticks_to_seconds(msg.time, current_tempo)

            if delta_seconds > 0:
                # Wait for the appropriate time before processing next event
                elapsed = time.time() - start_time
                sleep_time = current_time + delta_seconds - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            current_time += delta_seconds

            # Handle tempo changes
            if msg.type == "set_tempo":
                current_tempo = msg.tempo
                print(f"Tempo change: {60000000/current_tempo:.1f} BPM")

            # Handle note events
            elif msg.type == "note_on" and msg.velocity > 0:
                # Note on event - just record the start time
                active_notes[msg.note] = current_time
                frequency = note_to_frequency(msg.note)
                print(
                    f"Note ON: {msg.note} (freq: {frequency:.2f} Hz) at {current_time:.3f}s"
                )
                motor(frequency)

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                # Note off event - play the note with correct duration
                if msg.note in active_notes:
                    note_start_time = active_notes[msg.note]
                    note_duration = current_time - note_start_time

                    if note_duration > 0:  # Only play if duration is positive
                        frequency = note_to_frequency(msg.note)

                        print(f"Note OFF: {msg.note} (duration: {note_duration:.3f}s)")
                        motor()

                    del active_notes[msg.note]

        print("Done")

    except Exception as e:
        print(f"Error: {e}")


play_midi_track("portal.mid")
