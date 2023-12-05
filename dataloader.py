import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import requests
import os


class MusicDataset(Dataset):
    def __init__(self, config, max_note=88, min_note=21, split='train'):
        """The config is only the dataset part of the config file."""
        self.config = config
        self.max_note = max_note
        self.min_note = min_note
        self.path = config.path
        self.split = config.split
        
        self.data = self.load_process_data()
        self.sequence_lengths = self.data['sequence_lengths']
        self.encodings = self.data['encodings']
        self.masks = self.data['masks']
    
    def download_dataset(self):
        """
        Download a dataset from a given URL.

        Args:
        url (str): URL of the dataset to download.
        file_name (str): Name of the file to save the downloaded data.
        Note: By default, the data will be stored in data directory.
        """
        
        def download(file_name, url):
            response = requests.get(url)
            os.makedirs('data', exist_ok=True)
            
            if os.path.exists(os.path.join('data', file_name)):
                print(f"{file_name} already exists. Skipping download.")
                return
            
            if response.status_code == 200:
                with open(os.path.join('data', file_name) , 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded {file_name}")
            else:
                print(f"Failed to download {file_name}. Status code: {response.status_code}")
                
        for data_dict in self.config.urls:
            print(next(iter(data_dict.items())))
            file_name, url = next(iter(data_dict.items()))
            download(file_name, url)
                

    def read_pickle_from_url(self, path, split='train'):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data


    def load_process_data(self):
        """
        Data loader for the music dataset.
        split: train, test
        music: each music in the split_data: 
        keys: each set of note e.g (60, 64, 67) in the music
        notes: each note in e.g (60, 64, 67)
        Returns: a one-hot encoding of the music at the indecies of the notes, sequence length of each music
        """
        if self.config.download_first: #if dataset doesnt exist
            self.download_dataset()
        
        data_dict = {}
        
        data = self.read_pickle_from_url(self.path) #e.g 229 music data
        split_data = data[self.split]
        
        all_music_one_hot_list = []    
        sequence_lengths = []

        for music in split_data:
            one_hot_matrix = np.zeros((len(music), self.max_note), dtype=int)
            
            for row_index, keys in enumerate(music):

                for note in keys:
                    one_hot_matrix[row_index, note - self.min_note] = 1  
                    
            all_music_one_hot_list.append(one_hot_matrix)
            sequence_lengths.append(len(music))


        #pad music in all_music_one_hot with zeros until max_sequence_length with -1
        assert len(all_music_one_hot_list) == len(split_data)
        max_sequence_length = max(sequence_lengths)
        split_length = len(split_data)

        padded_all_music_one_hot = pad_sequence([torch.tensor(music) for music in all_music_one_hot_list], batch_first=True, padding_value=-1)

        masks = torch.zeros(( split_length, max_sequence_length, self.max_note ))
        masks = padded_all_music_one_hot != -1
    
        data_dict['encodings'] = padded_all_music_one_hot
        data_dict['sequence_lengths'] = sequence_lengths
        data_dict['masks'] = masks
        
        return data_dict
        
    def get_max_sequence_length(self):
        return max(self.sequence_lengths)
    
    def __getitem__(self, index):
        return self.encodings[index], self.masks[index], self.sequence_lengths[index]
        #encodings dim: (bs, seq_len, 88)
    def __len__(self):
        return len(self.encodings)
    
    
