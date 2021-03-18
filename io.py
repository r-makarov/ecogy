import numpy as np
import quantities as pq
import mne
import neo
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import copy

#-----------------------------------------------Probes-----------------------------------------------

class Probe():
    
       def __init__(self, df, name='Probe'):
        
        self.name = name
        self.df = df
        self.x = df.x.to_numpy()
        self.y = df.y.to_numpy()
        self.channels = df.channel.to_numpy()
        
        self.n_rows = np.unique(self.x).shape[0]
        self.n_cols = np.unique(self.y).shape[0]
        
        def save(self, path):
            assert path[-1] == '/'
            pd.to_csv(path + name + '.csv')
        
        
class LinearProbe(Probe):
    
    def __init__(self, df, name='A1x16'):
        super(LinearProbe, self).__init__(df, name)
        
        self.inserted_between = []
        
    def show(self):
    
        plt.figure(figsize=[3,8])
        
        plt.scatter(self.x, self.y, c='r', alpha=0.2, s=50)

        for i in range(len(self.channels)):
            plt.text(self.x[i], self.y[i], self.channels[i], ha='center', va='center')

      
        plt.ylim(self.y[-1]-100, 100)
        plt.xlim(-200, 200)
        plt.xticks([],[])
        
        plt.yticks(ticks=self.y, labels = self.y)
        
        plt.title(f'Probe_{self.name}_{np.abs(self.y[1])}um', loc='center')
        
        plt.tight_layout()
        
        
class FilmProbe(Probe):
    
    def __init__(self, df, name='XA60_FILM'):
        super(FilmProbe, self).__init__(df, name)
        
        self.boards = df.board.to_numpy()
        self.u_boards = np.sort(np.unique(self.boards))
        
    def show(self, amplitudes=[], ax=None):
    
        
        
        if ax:
            ax.clear()
            plt.sca(ax)
        else:
            plt.figure(figsize=[7.5,4.5])
        
        plt.plot(self.x, self.y, '.r')

        for i in range(len(self.channels)):
            plt.text(self.x[i], self.y[i], self.channels[i], ha='center', va='center')

      
        plt.xlim(-200, 3800)
        plt.ylim(-200, 2200)
        
        plt.xticks(ticks=np.arange(0,3700,400), labels = np.arange(1,11))
        plt.yticks(ticks=np.arange(0,2100,400), labels = list('FEDCBA'))
        
        plt.title(f'Probe_{self.name}_{self.u_boards[0]}_{self.u_boards[1]}', loc='left')
        plt.title('{neck:top, face:down}', loc='right')
        
        if amplitudes:
            plt.imshow(amplitudes.reshape(self.n_cols, self.n_rows), extent=[0, self.x.max(), 0, self.y.max()])
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        

#-----------------------------------------------Reader-----------------------------------------------

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
                


#------------------------------------------------Raw-------------------------------------------------

def get_raw(reader,
            selected_channels,
             block_index = 0, 
             seg_index = 0, 
             t_start = None, 
             t_stop = None):
    
    """
    Returns data from a Reader as a numpy array
    """
    
    #1. make a copy of the reader and select channels for a given probe
    _reader = copy.deepcopy(reader)
    _reader.select_channels(selected_channels)
    
    time_slice = (t_start, t_stop)
    
    seg = _reader.read_segment(block_index, 
                               seg_index,
                               time_slice=time_slice)
    
    analogsignal = seg.analogsignals[0]
    
    raw = mne.io.RawArray(analogsignal.T.magnitude, 
                          mne.create_info(_reader.df_ch['name'].to_list(), _reader.get_sfreq(), ch_types='eeg'))
    
    raw.reorder_channels(selected_channels)

    return raw



#-----------------------------------------------Events-----------------------------------------------

def get_events(reader,
               block_index = 0, 
               seg_index = 0, 
               t_start = None, 
               t_stop = None):
    
    #1. make a copy of the reader and leave only a single channel to speed up the function
    _reader = copy.deepcopy(reader)
    _reader.select_channels(_reader.df_ch['name'][0])
    
    
    #2. Get a Segment for a given time slice
    time_slice = (t_start, t_stop)
    
    seg = _reader.read_segment(block_index, 
                               seg_index,
                               time_slice=time_slice)
    
    #3. Create an DataFrame
    df_ev = pd.DataFrame(columns=['event_id', 'ttl', 'time', 'ts'])

    #4. For every type of events, for every event put its time in the dataframe
    
    for ev_type in seg.events:
        print(f'{ev_type.name} {len(ev_type.magnitude)}')
    
        for e in ev_type.magnitude:
        
            idx = len(df_ev.index) #the index of the row where information about the event will be placed
        
            
            df_ev.loc[idx, 'event_id'] = int(ev_type.name[16:18])
            df_ev.loc[idx, 'ttl'] = int(ev_type.name[23:])
            df_ev.loc[idx, 'time'] = e                                           #time in seconds
            df_ev.loc[idx, 'ts'] = np.round(e*_reader.get_sfreq()).astype('int') #convert time to timesteps
    
    df_ev.sort_values(by='time', inplace=True)
    df_ev.reset_index(drop=True, inplace=True)

    return df_ev


#-----------------------------------------------Epochs-----------------------------------------------

def get_epochs(reader, 
               selected_channels, 
               events,
               block_index=0, seg_index=0,
               tmin=-0.2, tmax=0.5):
    
    """
    Uploades slices of raw data and converts them to epochs one by one,
    instead of passing the whole raw data array as an input to mne.Epochs()
    """
    
    #1. make a copy of the reader and select channels for a given probe
    _reader = copy.deepcopy(reader)
    _reader.select_channels(selected_channels)
    
    
    sfreq = reader.get_sfreq()
    tmax -= 1/sfreq #to avoide odd-number epoch length
    
    #2. convert slices of raw data one by one
    epochs_list = []
    for i, ev in tqdm(enumerate(events)):
        
        # expand time slice to 0.1 s from both sides to prevent boundary artifacts
        t_start,t_stop = [ev[0]/sfreq + tmin - 0.1, ev[0]/sfreq + tmax + 0.1] 
        
        # get a slice of raw data
        _raw = get_raw(reader,
                       selected_channels, 
                       block_index, seg_index,
                       t_start, t_stop)
        
        # convert the slice to an epoch
        epoch = mne.Epochs(_raw, 
                           np.array([np.abs(tmin-0.1)*sfreq,0,0]).astype('int').reshape(1,-1), 
                           tmin=tmin, 
                           tmax=tmax,
                           preload=True)
        
        epochs_list.append(epoch)
        
    epochs = mne.concatenate_epochs(epochs_list)
    
    return epochs


def remove_artifact(epochs, fill_len=0.005, fill_with=0):
    epochs._data[:,:,np.abs(epochs.times) < fill_len] = fill_with



#------------------------------------------------Misc------------------------------------------------
    
def get_time_slice(reader, t_start, t_stop, block_index, seg_index):
    
    if t_start is None:
        t_start = reader.get_signal_t_start(block_index, seg_index)
    else:
        t_start = t_start
        
    if t_stop is None:
        t_stop = (reader._sigs_t_stop[seg_index] - reader.global_t_start)
    else:
        t_stop = t_stop
        
    print('{:<30}'.format(f'Starts at'), ':', t_start, 's')
    print('{:<30}'.format(f'Stops at'), ':', t_stop, 's')
    
    return (t_start, t_stop)