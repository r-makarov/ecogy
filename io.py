import numpy as np
import quantities as pq
import mne
import neo
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Probes

class Probe():
    
       def __init__(self, df, name='Probe'):
        
        self.name = name
        self.x = df.x.to_numpy()
        self.y = df.y.to_numpy()
        self.channels = df.channel.to_numpy()
        self.boards = df.board.to_numpy()
        self.u_boards = np.sort(np.unique(self.boards))
        
class FilmProbe(Probe):
    
    def __init__(self, df, name='XA60_FILM'):
        super(FilmProbe, self).__init__(df, name)
        
    def show(self):
    
        plt.figure(figsize=[7.5,4.5])
        
        plt.scatter(self.x, self.y, c='r', alpha=0.2, s=50)

        for i in range(len(self.channels)):
            plt.text(self.x[i], self.y[i], self.channels[i], ha='center', va='center')

      
        plt.xlim(-200, 3800)
        plt.ylim(-200, 2200)
        
        plt.xticks(ticks=np.arange(0,3700,400), labels = np.arange(1,11))
        plt.yticks(ticks=np.arange(0,2100,400), labels = list('FEDCBA'))
        
        plt.title(f'Probe_{self.name}_{self.u_boards[0]}_{self.u_boards[1]}', loc='left')
        plt.title('{neck:top, face:down}', loc='right')
        
        plt.tight_layout()
        

# Reader

class Reader(neo.io.NeuralynxIO):
    
    def __init__(self, dirname, use_cache=False, cache_path='same_as_resource'):
        super(Reader, self).__init__(dirname, use_cache=False, cache_path='same_as_resource')
        
        self.df_ch = pd.DataFrame(self.header['signal_channels'])
        self.df_ch.insert(0, 'type', self.df_ch['name'].apply(lambda x:x[:3]))
        
    def select_channels(self, selected_channels):
        
        self.df_ch = self.df_ch[self.df_ch['name'].apply(lambda ch: ch in selected_channels)]
        self.header['signal_channels'] = np.array(self.df_ch.iloc[:,1:].to_records(index=False))
        
    def get_sfreq(self):
        unique_rates = self.df_ch['sampling_rate'].unique()
        assert len(unique_rates)==1, "There are channels with different sampling rate!"
        return unique_rates.astype(int)[0]
    
    def info(self):
        print('{:<30}'.format('Total number of channels'), 
              ':',
              self.signal_channels_count())
        
        for t in self.df_ch['type'].unique():
            print('{:<30}'.format(f'    Number of {t} channels'),
                  ':', 
                  self.df_ch["type"].apply(lambda x:x == t).sum())
        
        print('\n')
        print('{:<30}'.format('Sampling frequency'), ':',self.get_sfreq(), 'Hz\n')
        
        for b in range(self.block_count()):
            print('{:<30}'.format(f'Block {b} has:'))
            for s in range(self.segment_count(b)):
                print('{:<30}'.format(f'    Segment {s} of length'), ':', 
                      self.get_signal_size(b,s)/self.get_sfreq(), 's (', 
                      self.get_signal_size(b,s)/self.get_sfreq()/60, ')min')
                
    
        
    
    
# Raw

def get_time_slice(t_start, t_stop, block_index, seg_index):
    
    if t_start is None:
        t_start = reader.get_signal_t_start(block_index, seg_index)
    else:
        t_start = t_start
        
    if t_stop is None:
        t_stop = (reader._sigs_t_stop[seg_index] - reader.global_t_start)
    else:
        t_stop = t_stop
        
    return (t_start, t_stop)



def get_raw(reader,
            selected_channels,
             block_index = 0, 
             seg_index = 0, 
             t_start = None, 
             t_stop = None):
    
    """
    Returns data from a Reader as a numpy array
    """
    
    
    reader.select_channels(selected_channels)
    reader.info()
    
    time_slice = get_time_slice(t_start,
                            t_stop, 
                            block_index, 
                            seg_index)
    
    seg = reader.read_segment(block_index, 
                          seg_index,
                          time_slice=time_slice)
    
    analogsignal = seg.analogsignals[0]
    
    raw = mne.io.RawArray(analogsignal.T.magnitude, 
                          mne.create_info(reader.df_ch['name'].to_list(), reader.get_sfreq(), ch_types='eeg'))
    
    raw.reorder_channels(selected_channels)
    
    
    
    return raw