import torch
import pickle
import torch
import numpy as np
from midiutil import MIDIFile
from pydub import AudioSegment
from midi2audio import FluidSynth
import os

def midi_to_song(data, file_name, experiment_name, mp3=False):
    fs = FluidSynth('default.sf2')
    # Create a MIDI file
    midi_file = MIDIFile(1)  # One track
    track = 0
    time = 0  # Start at the beginning
    midi_file.addTrackName(track, time, "Track")
    midi_file.addTempo(track, time, 120)  # Set the tempo

    # Add notes to the MIDI file
    for chord in data:  # Assuming you're using the first song
        duration = 1  # Duration of each chord, you can change this
        for note in chord:
            midi_file.addNote(track, 0, note, time, duration, 100)  # channel set to 0, volume 100
        time += duration

    # Write the MIDI file to disk
    with open(f"{file_name}.mid", "wb") as output_file:
        midi_file.writeFile(output_file)

    path = os.path.join(experiment_name, f'{file_name}.wav')
    os.makedirs(path, exist_ok=True)
    
    # Synthesize MIDI to WAV using FluidSynth
    # fs = FluidSynth('path_to_soundfont.sf2')
    fs.midi_to_audio(f'{file_name}.mid', f'{file_name}.wav')

    # # Convert WAV to MP3
    if mp3:
        audio = AudioSegment.from_wav(f'{file_name}.wav')
        audio.export(f'{file_name}.mp3', format='mp3')
    
    