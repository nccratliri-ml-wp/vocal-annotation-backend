import argparse
import json
from flask import Flask, jsonify, abort, make_response, request, Response, send_file
from flask_cors import CORS
import numpy as np
import matplotlib.cm as cm
import librosa
import soundfile as sf
from PIL import Image
from skimage.transform import resize
import threading
import base64
import io
import hashlib
from scipy import signal
import random
from uuid import uuid4

import requests
from io import BytesIO
from pydub import AudioSegment

from spec_utils import SpecCalConstantQ, SpecCalLogMel
from segmentation_utils import call_segment_service

# Make Flask application
app = Flask(__name__)
CORS(app)

def load_audio_from_url(audio_url):
    response = requests.get(audio_url)
    response.raise_for_status()
    audio_file = BytesIO(response.content)

    audio_data = AudioSegment.from_file(audio_file)
    wav_bytes = BytesIO()
    ## always convert to wav, which can be safely loaded via librosa
    audio_data.export(wav_bytes, format="wav")
    wav_bytes.seek(0)  # Reset the pointer of the BytesIO object
    # Load the WAV bytes with librosa
    audio, sr = librosa.load(wav_bytes, sr=None, mono=False)  # 'sr=None' to preserve the original sampling rate
    return audio, sr

def compute_md5(byte_stream):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()
    # Update the hash object with the byte stream
    md5_hash.update(byte_stream)

    # Get the hexadecimal representation of the hash
    return md5_hash.hexdigest()

def bytes_to_base64_string(f_bytes):
    return base64.b64encode(f_bytes).decode('ASCII')

def base64_string_to_bytes(base64_string):
    return base64.b64decode(base64_string)

def get_spectrogram( audio, sr, start_time, clip_duration, 
                     num_spec_columns = 1000, 
                     min_frequency = None, max_frequency = None, n_bins = 256,
                     spec_cal_method = "log-mel",
                     n_fft = None,
                     bins_per_octave = None
                   ):
    
    hop_length = int( clip_duration * sr / num_spec_columns )
    if spec_cal_method == "log-mel":
        spec_cal = SpecCalLogMel( sr = sr, hop_length = hop_length, 
                                  min_frequency = min_frequency, max_frequency = max_frequency,
                                  n_bins = n_bins, n_fft = n_fft )
    elif spec_cal_method == "constant-q":
        spec_cal = SpecCalConstantQ( sr = sr, hop_length = hop_length, 
                                  min_frequency = min_frequency, max_frequency = max_frequency,
                                  n_bins = n_bins, bins_per_octave = bins_per_octave )
    else:
        assert False, "Unsupported spectrogram computation method!"
    
    start_time = max( start_time, 0.0 )    
    audio_clip = audio[ int( start_time * sr ): int( start_time * sr ) + int( clip_duration * sr ) ]
    audio_clip = np.concatenate( [ audio_clip, np.zeros( int( clip_duration * sr ) - len(audio_clip) ) ], axis = 0 )
    audio_clip = audio_clip.astype(np.float32)
        
    log_mel_spec = spec_cal( audio_clip )
    
    ## resize the log_mel_spec
    log_mel_spec = resize( log_mel_spec, ( n_bins, num_spec_columns, 3 ) )
    return log_mel_spec, spec_cal.freqs

def min_max_downsample(array, rough_target_length):
    
    half_target_length = rough_target_length // 2
    downsample_factor = int(np.round( len(array) / half_target_length ))
    
    # Calculate the size of the downsampled array
    downsampled_length = len(array) // downsample_factor
    remainder = len(array) % downsample_factor

    # Handle case where array size is not a multiple of downsample_factor
    if remainder != 0:
        # Pad array to make it fit exactly into the downsample factor
        padded_size = downsampled_length * downsample_factor + downsample_factor
        padded_array = np.pad(array, (0, padded_size - array.size), 'constant', constant_values=np.nan)
    else:
        padded_array = array

    # Reshape the array to group by downsample factor
    reshaped_array = padded_array.reshape(-1, downsample_factor)

    # Calculate min and max for each group
    mins = np.nanmin(reshaped_array, axis=1)
    maxs = np.nanmax(reshaped_array, axis=1)

    # Combine mins and maxs into a single array
    downsampled_array = np.stack((mins, maxs), axis=-1)
    # If there was a remainder, trim the last element (as it was padded)
    if remainder != 0:
        downsampled_array = downsampled_array[:-1]
    
    downsampled_array = downsampled_array.flatten()
    return downsampled_array

def resample_audio( audio, target_length = 100000 ):
    if len(audio) <= target_length:
        sampled_audio = audio
    else:
        sampled_audio = min_max_downsample(audio, 10000)
        
    if len(sampled_audio) == 0:
        final_audio = np.zeros( target_length )
    else:
        final_audio = sampled_audio

    final_audio = final_audio.astype(np.float32)
    return final_audio
    
def register_new_audio( audio, sr, audio_id ): 
    global audio_dict
    audio_dict[audio_id] = {
        "audio":audio,
        "sr":sr,
        "percentile_up":np.percentile(audio, 99.99),
        "percentile_down":np.percentile(audio, 0.01)
    }   

@app.route("/upload", methods=['POST'])
def upload():
    global audio_dict, num_spec_columns, n_bins
    
    newAudioFile = request.files['newAudioFile']
    audio_multi_channels, sr = librosa.load(newAudioFile, sr = None, mono = False )   
    if len( audio_multi_channels.shape ) == 1:
        audio_multi_channels = audio_multi_channels[ np.newaxis,: ]
    
    spec_all_channels = []
    for pos in range( audio_multi_channels.shape[0] ):
        audio = audio_multi_channels[pos]

        audio_id = str( uuid4() )
        print(audio_id)
        register_new_audio( audio, sr, audio_id )
        
        whole_audio_spec, freqs = get_spectrogram( audio, sr, 0, len( audio ) / sr, 
                        num_spec_columns = num_spec_columns, 
                        n_bins = n_bins,
                        spec_cal_method = "log-mel"
                    )
        
        spec_3d_arr = np.asarray(whole_audio_spec)
        spec_3d_arr = np.minimum(spec_3d_arr * 255, 255).astype(np.uint8)
        im = Image.fromarray(spec_3d_arr)
        # Create an in-memory binary stream
        buffer = io.BytesIO()
        # Save the image to the stream
        im.save(buffer, format="JPEG")
        # Get the content of the stream and encode it to base64
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode()
        
        spec_all_channels.append( {"spec":base64_string,
                "freqs":freqs.tolist(),
                "audio_duration": len( audio ) / sr,
                "audio_id": audio_id
            } )
        
    return jsonify(spec_all_channels), 201

@app.route("/upload-by-url", methods=['POST'])
def upload_by_url():
    global audio_dict, num_spec_columns, n_bins
    request_info = request.json
    audio_url = request_info['audio_url']

    audio_multi_channels, sr = load_audio_from_url(audio_url)
    if len( audio_multi_channels.shape ) == 1:
        audio_multi_channels = audio_multi_channels[ np.newaxis,: ]

    spec_all_channels = []
    for pos in range( audio_multi_channels.shape[0] ):
        audio = audio_multi_channels[pos]
    
        audio_id = str( uuid4() )
        print(audio_id)
        register_new_audio( audio, sr, audio_id )
        
        whole_audio_spec, freqs = get_spectrogram( audio, sr, 0, len( audio ) / sr, 
                        num_spec_columns = num_spec_columns, 
                        n_bins = n_bins,
                        spec_cal_method = "log-mel"
                    )
        
        spec_3d_arr = np.asarray(whole_audio_spec)
        spec_3d_arr = np.minimum(spec_3d_arr * 255, 255).astype(np.uint8)
        im = Image.fromarray(spec_3d_arr)
        # Create an in-memory binary stream
        buffer = io.BytesIO()
        # Save the image to the stream
        im.save(buffer, format="JPEG")
        # Get the content of the stream and encode it to base64
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode()
        
        spec_all_channels.append( {"spec":base64_string,
                "freqs":freqs.tolist(),
                "audio_duration": len( audio ) / sr,
                "audio_id": audio_id
            } )
        
    return jsonify(spec_all_channels), 201
    
@app.route("/get-audio-clip-spec", methods=['POST'])
def get_audio_clip_spec():
    global audio_dict, num_spec_columns, n_bins
    
    request_info = request.json

    print(request_info.keys())

    audio_id = request_info["audio_id"]
    start_time = request_info["start_time"]
    clip_duration = request_info["clip_duration"]
    
    audio = audio_dict[audio_id]["audio"]
    sr = audio_dict[audio_id]["sr"]
    
    min_frequency = request_info.get( "min_frequency", 0 )
    max_frequency = request_info.get( "max_frequency", sr//2 )
    max_frequency = min( max_frequency, sr//2 )
    
    spec_cal_method = request_info.get( "spec_cal_method", "log-mel" )
    
    try:
        n_fft = int(request_info.get("n_fft", None))
    except:
        n_fft = None
    try:
        bins_per_octave = int(request_info.get("bins_per_octave", None))
    except:
        bins_per_octave = None

    if spec_cal_method == "log-mel":
        print("Computing LogMel, n_fft =", n_fft)
    elif spec_cal_method == "constant-q":
        print("Computing Constant-Q, bins_per_octave=", bins_per_octave)

    audio_clip_spec, freqs = get_spectrogram( audio, sr, start_time, clip_duration, 
                     num_spec_columns = num_spec_columns, 
                     min_frequency = min_frequency, max_frequency = max_frequency, n_bins = n_bins,
                     spec_cal_method = spec_cal_method,
                     n_fft = n_fft,
                     bins_per_octave = bins_per_octave
                   )
    
    spec_3d_arr = np.asarray(audio_clip_spec)
    spec_3d_arr = np.minimum(spec_3d_arr * 255, 255).astype(np.uint8)
    im = Image.fromarray(spec_3d_arr)

    # Create an in-memory binary stream
    buffer = io.BytesIO()
    # Save the image to the stream
    im.save(buffer, format="JPEG")
    # Get the content of the stream and encode it to base64
    base64_bytes = base64.b64encode(buffer.getvalue())
    base64_string = base64_bytes.decode()
    
    buffer.seek(0)
    
    return {"spec":base64_string, 
            "freqs": freqs.tolist()}

@app.route("/get-audio-clip-wav", methods=['POST'])
def get_audio_clip_wav():
    global audio_dict
    
    request_info = request.json
    audio_id = request_info["audio_id"]
    start_time = request_info["start_time"]
    clip_duration = request_info["clip_duration"]
    
    audio = audio_dict[audio_id]["audio"]
    sr = audio_dict[audio_id]["sr"]
    
    audio_clip = audio[ int( start_time * sr ): int( start_time * sr ) + int(clip_duration * sr) ] # int( (start_time + clip_duration) * sr ) ]
    ## always pad the audio to the specified length of duration
    audio_clip = np.concatenate( [ audio_clip, np.zeros( int(clip_duration * sr) - len(audio_clip) ) ], axis = 0 ).astype(np.float32)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_clip, samplerate=sr, format='WAV')
    base64_bytes = base64.b64encode(buffer.getvalue())
    base64_string = base64_bytes.decode()
    return {"wav":base64_string}

@app.route("/get-audio-clip-for-visualization", methods=['POST'])
def get_audio_clip_for_visualization():
    global audio_dict
    
    request_info = request.json
    audio_id = request_info["audio_id"]
    start_time = request_info["start_time"]
    clip_duration = request_info["clip_duration"]
    target_length = request_info.get("target_length", 100000)
    
    audio = audio_dict[audio_id]["audio"]
    sr = audio_dict[audio_id]["sr"]
    percentile_up = audio_dict[audio_id]["percentile_up"]
    percentile_down = audio_dict[audio_id]["percentile_down"]
    
    audio_clip = audio[ int( start_time * sr ):int( (start_time + clip_duration) * sr ) ]
    audio_clip = resample_audio( audio_clip, target_length )

    audio_clip = np.clip( audio_clip, a_min = percentile_down, a_max =  percentile_up )
    return jsonify({"wav_array":audio_clip.tolist()}), 201

@app.route("/get-labels", methods=['POST'])
def get_labels():
    global audio_dict, args
    
    request_info = request.json
    audio_id = request_info["audio_id"]
    
    audio = audio_dict[audio_id]["audio"]
    sr = audio_dict[audio_id]["sr"]
    
    min_frequency = request_info.get( "min_frequency", None )
    spec_time_step = request_info.get( "spec_time_step", None )
    min_segment_length = request_info.get( "min_segment_length", None )
    eps = request_info.get( "eps", None )
    
    prediction = call_segment_service( args.segmentation_service_address, 
                          audio,
                          sr,
                          min_frequency = min_frequency,
                          spec_time_step = spec_time_step,
                          min_segment_length = min_segment_length,
                          eps = eps,
                          num_trials = 1
                        )
    return jsonify({"labels":prediction}), 201


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-flask_port", help="The port of the flask app.", default=8050, type=int)
    parser.add_argument("-segmentation_service_address", help="The address to the WhisperSeg segmentation API.", default="https://07bd-130-60-24-13.ngrok-free.app/segment")
    args = parser.parse_args()
    
    audio_dict = {}
    num_spec_columns = 1000
    n_bins = 200
    
    print( "Waiting for requests..." )

    app.run( host='0.0.0.0', port=args.flask_port, debug = True )