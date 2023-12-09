import torch
import pickle
import torch
import numpy as np
from midiutil import MIDIFile
from pydub import AudioSegment
from midi2audio import FluidSynth
from torch.nn.utils.rnn import pad_sequence
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
    with open(os.path.join(experiment_name, f"{file_name}.mid"), "wb") as output_file:
        midi_file.writeFile(output_file)

    os.makedirs(experiment_name, exist_ok=True)
    
    # Synthesize MIDI to WAV using FluidSynth
    # fs = FluidSynth('path_to_soundfont.sf2')
    fs.midi_to_audio(os.path.join(experiment_name, f'{file_name}.mid'), 
                     os.path.join(experiment_name, f'{file_name}.wav'))

    # # Convert WAV to MP3
    if mp3:
        audio = AudioSegment.from_wav(os.path.join(experiment_name, f'{file_name}.wav'))
        audio.export(os.path.join(experiment_name, f'{file_name}.mp3'), format='mp3')
    
    

def log_midis(midi, orig_midi):
    if orig_midi:
        for i, (midi_key, midi_orig) in enumerate(zip(midi, orig_midi)):
            print(f' recon_midi: {midi_key}')
            print(f' orig_midi: {midi_orig}')
            print('-------------------')
            
    else:
        for i, midi_key  in enumerate(midi):
            print(f' recon_midi: {midi_key}')
            print('-------------------')
            



def collate_fn(batch):
    sequences, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths)
    return sequences_padded, lengths