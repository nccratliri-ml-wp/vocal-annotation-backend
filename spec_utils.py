import os
from glob import glob
import librosa
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import sys
from librosa.core.convert import cqt_frequencies, mel_frequencies

class SpecCalConstantQ:
    def __init__(self, sr, hop_length, min_frequency = None, max_frequency = None, n_bins = 256, 
                 bins_per_octave = None,
                 color_map = "inferno", **kwargs ):
        self.sr = sr
        self.hop_length = hop_length
        self.min_frequency = 50 if min_frequency is None else min_frequency
        if self.min_frequency <= 0:
            self.min_frequency = 50
        self.max_frequency = int(sr/2) if (max_frequency is None or max_frequency <= 0) else max_frequency
        if self.max_frequency > int(sr/2):
            self.max_frequency = int(sr/2)
        self.n_bins = n_bins
        
        ## derived fmax = fmin * (2**( n_bins/bins_per_octave ) ), we need to incrementally increase bins_per_octave and make sure derived fmax do not exceed Nyquist sampling rate
        # Initialize min_bins_per_octave to 1
        min_bins_per_octave = int(np.ceil(self.n_bins / np.log2(sys.float_info.max / self.min_frequency )))   ## this may be reffered as the Q value?
        # Incrementally increase min_bins_per_octave until the derived maximum frequency is below the Nyquist frequency
        while self.min_frequency * (2**(n_bins/min_bins_per_octave)) > self.max_frequency:
            min_bins_per_octave += 1
        self.min_bins_per_octave = min_bins_per_octave
        
        if bins_per_octave is None:
            self.bins_per_octave = self.min_bins_per_octave
        else:
            self.bins_per_octave = max( self.min_bins_per_octave, bins_per_octave )
        
        self.cmap = matplotlib.colormaps.get_cmap(color_map)
        self.freqs = np.round(np.array( cqt_frequencies( self.n_bins, fmin = self.min_frequency, bins_per_octave = self.bins_per_octave) ))

    def sigmoid(self, x, offset, low, high, sharpness):
        remapped = (x - offset) * sharpness
        y = remapped / (1 + np.abs(remapped))
    
        # guarantee the output starts at 0 and ends at 1
        x_high = (high - offset) * sharpness
        x_low = (low - offset) * sharpness
        y_high = x_high / (1 + np.abs(x_high))
        y_low = x_low / (1 + np.abs(x_low))
        y = (y - y_low) / (y_high - y_low)
    
        return y
    
    def min_max_norm(self, im, norm_min_value = None, norm_max_value = None):
        if norm_min_value is not None and norm_max_value is not None:
            pseudo_min = norm_min_value
            pseudo_max = norm_max_value
        else:
            pseudo_min = np.percentile(im, 0.01)
            pseudo_max = np.percentile(im, 99.99)
    
        # return (im - im.min()) / max(im.max() - im.min(), 1e-12)
        return np.clip( (im-pseudo_min) / max( pseudo_max - pseudo_min, 1e-12 ), 0, 1)
    
    def __call__(self, audio, norm_min_value = None, norm_max_value = None, return_min_max_spec_values = False ):
        cqt = librosa.cqt(audio,
                  sr=self.sr,
                  hop_length=self.hop_length,
                  fmin=self.min_frequency,
                  n_bins=self.n_bins,
                  bins_per_octave=self.bins_per_octave,
                  filter_scale=1.0,
                  window='blackmanharris')

        cqt_db = librosa.amplitude_to_db(np.abs(cqt))
        # cqt_db_norm = np.copy(cqt_db)
        # cqt_db_norm -= cqt_db_norm.min()
        # cqt_db_norm /= cqt_db_norm.max()

        min_spec_values = np.percentile(cqt_db, 0.01)
        max_spec_values = np.percentile(cqt_db, 99.99)

        cqt_db_norm = self.min_max_norm( cqt_db, norm_min_value, norm_max_value )
        
        result = self.sigmoid(cqt_db_norm, 0.5, 0, 1, 2) # cqt_db_norm
        result = result[:,:-1]
        result = result.astype(np.float32)
        
        spec = np.flip(self.cmap( result )[:,:,:3], axis = 0)
        
        if return_min_max_spec_values:
            return spec, min_spec_values, max_spec_values
        else:
            return spec
    
class SpecCalLogMel:
    def __init__(self, sr, hop_length, min_frequency = None, max_frequency = None, n_bins = 256,
                 n_fft = None,
                 color_map = "inferno", **kwargs ):
        self.sr = sr
        self.hop_length = hop_length
        self.min_frequency = 0 if min_frequency is None or min_frequency < 0 else min_frequency
        self.max_frequency = int(sr/2) if max_frequency is None or max_frequency <= 0 else max_frequency
        if self.max_frequency > int(sr/2):
            self.max_frequency = int(sr/2)
        self.n_bins = n_bins        
        self.cmap = matplotlib.colormaps.get_cmap(color_map)
        
        if n_fft is None or n_fft <= 0:
            if sr <= 32000:
                n_fft = 512
            elif sr <= 80000:
                n_fft = 1024
            elif sr <= 150000:
                n_fft = 2048
            elif sr <= 300000:
                n_fft = 4096
            else:
                n_fft = 8192
        else:
            n_fft = max(5, n_fft)
        self.n_fft = n_fft
        self.freq_upsampling_ratio = 8

        self.freqs = np.round(np.array( mel_frequencies( self.n_bins, fmin = self.min_frequency, fmax = self.max_frequency ) ))

    def sigmoid(self, x, offset, low, high, sharpness):
        remapped = (x - offset) * sharpness
        y = remapped / (1 + np.abs(remapped))
    
        # guarantee the output starts at 0 and ends at 1
        x_high = (high - offset) * sharpness
        x_low = (low - offset) * sharpness
        y_high = x_high / (1 + np.abs(x_high))
        y_low = x_low / (1 + np.abs(x_low))
        y = (y - y_low) / (y_high - y_low)

        return y
    
    def min_max_norm(self, im, norm_min_value = None, norm_max_value = None):
        if norm_min_value is not None and norm_max_value is not None:
            pseudo_min = norm_min_value
            pseudo_max = norm_max_value
        else:
            pseudo_min = np.percentile(im, 0.01)
            pseudo_max = np.percentile(im, 99.99)
    
        # return (im - im.min()) / max(im.max() - im.min(), 1e-12)
        return np.clip( (im-pseudo_min) / max( pseudo_max - pseudo_min, 1e-12 ), 0, 1)
    
    
    def __call__(self, audio, norm_min_value = None, norm_max_value = None, return_min_max_spec_values = False ):
        stft_result = librosa.stft( y=audio, hop_length=self.hop_length, n_fft=self.n_fft )[:,:-1]    
        spec = np.abs(stft_result)**2
    
        new_spec = np.zeros( ( spec.shape[0]*self.freq_upsampling_ratio, spec.shape[1] ) )
        for offset in range( self.freq_upsampling_ratio ):
            new_spec[ offset::self.freq_upsampling_ratio,: ] = spec
        pseudo_n_fft = self.n_fft * self.freq_upsampling_ratio
        new_spec = new_spec[ : pseudo_n_fft//2+1 ]
        melfb = librosa.filters.mel(sr=self.sr, n_mels=self.n_bins, n_fft = pseudo_n_fft, fmin=self.min_frequency, fmax = self.max_frequency )
        mel_spec = np.matmul( melfb, new_spec )

        log_mel_spec = librosa.power_to_db( mel_spec, ref=np.max )

        min_spec_values = np.percentile(log_mel_spec, 0.01)
        max_spec_values = np.percentile(log_mel_spec, 99.99)

        log_mel_spec = self.min_max_norm( log_mel_spec, norm_min_value, norm_max_value )
        
        log_mel_spec = np.flip(self.cmap( log_mel_spec )[:,:,:3], axis = 0)
    
        if return_min_max_spec_values:
            return log_mel_spec, min_spec_values, max_spec_values
        else:
            return log_mel_spec
    
class SpecCalDummy:
    def __init__(self, sr, hop_length, min_frequency = None, max_frequency = None, n_bins = 256,
                 n_fft = None,
                 color_map = "inferno", **kwargs ):
        self.sr = sr
        self.hop_length = hop_length
        self.min_frequency = 0 if min_frequency is None or min_frequency < 0 else min_frequency
        self.max_frequency = int(sr/2) if max_frequency is None or max_frequency <= 0 else max_frequency
        if self.max_frequency > int(sr/2):
            self.max_frequency = int(sr/2)
        self.n_bins = n_bins        
        self.cmap = matplotlib.colormaps.get_cmap(color_map)
        
        if n_fft is None or n_fft <= 0:
            if sr <= 32000:
                n_fft = 512
            elif sr <= 80000:
                n_fft = 1024
            elif sr <= 150000:
                n_fft = 2048
            elif sr <= 300000:
                n_fft = 4096
            else:
                n_fft = 8192
        else:
            n_fft = max(5, n_fft)
        self.n_fft = n_fft
        self.freq_upsampling_ratio = 8

        self.freqs = np.round(np.array( mel_frequencies( self.n_bins, fmin = self.min_frequency, fmax = self.max_frequency ) ))

    def sigmoid(self, x, offset, low, high, sharpness):
        remapped = (x - offset) * sharpness
        y = remapped / (1 + np.abs(remapped))
    
        # guarantee the output starts at 0 and ends at 1
        x_high = (high - offset) * sharpness
        x_low = (low - offset) * sharpness
        y_high = x_high / (1 + np.abs(x_high))
        y_low = x_low / (1 + np.abs(x_low))
        y = (y - y_low) / (y_high - y_low)

        return y
    
    def min_max_norm(self, im):
        ## This is used to increase the contrast of the image
        pseudo_min = np.percentile(im, 0.01)
        pseudo_max = np.percentile(im, 99.99)
    
        # return (im - im.min()) / max(im.max() - im.min(), 1e-12)
        return (im-pseudo_min) / max( pseudo_max - pseudo_min, 1e-12 )
    

    def get_log_mel_spec_uncolored(self, audio, hop_length, n_fft ):
        stft_result = librosa.stft( y=audio, hop_length=hop_length, n_fft=n_fft )[:,:-1]    
        spec = np.abs(stft_result)**2
    
        new_spec = np.zeros( ( spec.shape[0]*self.freq_upsampling_ratio, spec.shape[1] ) )
        for offset in range( self.freq_upsampling_ratio ):
            new_spec[ offset::self.freq_upsampling_ratio,: ] = spec
        pseudo_n_fft = n_fft * self.freq_upsampling_ratio
        new_spec = new_spec[ : pseudo_n_fft//2+1 ]
        melfb = librosa.filters.mel(sr=self.sr, n_mels=self.n_bins, n_fft = pseudo_n_fft, fmin=self.min_frequency, fmax = self.max_frequency )
        mel_spec = np.matmul(melfb, new_spec)

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = self.min_max_norm( log_mel_spec )
        
        return log_mel_spec

    def get_dummy_spec_uncolored( self, audio, hop_length ):
        try:
            log_mel_spec = audio != 0
            log_mel_spec_new = np.logical_or(log_mel_spec, np.concatenate( [ np.zeros( self.n_fft // 2 ), log_mel_spec[:-self.n_fft // 2] ], axis = 0 ) )
            log_mel_spec = np.logical_or(log_mel_spec_new, np.concatenate( [ log_mel_spec[self.n_fft // 2:], np.zeros( self.n_fft // 2 )  ], axis = 0 ) )
        except:
            log_mel_spec = audio != 0
            
        log_mel_spec = log_mel_spec[::hop_length][np.newaxis, :]
        log_mel_spec = np.repeat( log_mel_spec, self.n_bins, axis = 0 ).astype(np.float32)
        
        return log_mel_spec        
    
    # def __call__(self, audio, hop_length = None, n_fft = None ):
        # if hop_length is None:
        #     hop_length = self.hop_length
        # if n_fft is None:
        #     n_fft = min( len(audio), 16000 )

    def __call__(self, audio, norm_min_value = None, norm_max_value = None, return_min_max_spec_values = False ):
        hop_length = self.hop_length
        n_fft = min( len(audio), self.n_fft )
            
        log_mel_spec = self.get_log_mel_spec_uncolored( audio, hop_length, n_fft )
        dummy_spec = self.get_dummy_spec_uncolored( audio, hop_length )
        
        mix_shape = np.minimum( dummy_spec.shape, log_mel_spec.shape )
        log_mel_spec = log_mel_spec[:mix_shape[0], :mix_shape[1] ]
        dummy_spec = dummy_spec[:mix_shape[0], :mix_shape[1] ]
             
        spec_row = dummy_spec[0,:]
        spec_row = np.array([0] + spec_row.tolist() + [0])
        diff = spec_row - np.array( [0] + spec_row[:-1].tolist() ) 
        onsets = np.argwhere( diff > 0 )[:,0] - 1
        offsets = np.argwhere( diff < 0 )[:,0] - 1
        for onset, offset in zip( onsets, offsets ):
            log_mel_spec_block = log_mel_spec[ :, onset:offset ]
            mask_indices = log_mel_spec_block.mean(axis = 1) < 0.6
            mask_indices = np.logical_or( mask_indices, np.array(mask_indices[1:].tolist() + [0]))
            inverse_mask_indices = mask_indices == 0
            dummy_spec[mask_indices, onset:offset] = 0
        color_spec = np.flip(self.cmap( dummy_spec )[:,:,:3], axis = 0)
        if return_min_max_spec_values:
            return color_spec, None, None
        else:
            return color_spec
    
