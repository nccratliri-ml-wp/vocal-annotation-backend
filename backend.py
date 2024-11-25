import argparse
import json
from flask import Flask, jsonify, abort, make_response, request, Response, send_file
from flask_cors import CORS
import matplotlib.pyplot as plt
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
from datetime import datetime, timedelta
import time
import threading
from spec_utils import SpecCalConstantQ, SpecCalLogMel, SpecCalDummy
from segmentation_utils import *
import re
import zipfile
from urllib.parse import quote

# Make Flask application
app = Flask(__name__)
CORS(app)

def load_audio_from_url(audio_url, sr = None):
    response = requests.get(audio_url)
    response.raise_for_status()
    audio_file = BytesIO(response.content)

    audio_data = AudioSegment.from_file(audio_file)
    wav_bytes = BytesIO()
    ## always convert to wav, which can be safely loaded via librosa
    audio_data.export(wav_bytes, format="wav")
    wav_bytes.seek(0)  # Reset the pointer of the BytesIO object
    # Load the WAV bytes with librosa
    
    orig_audio, orig_sr = librosa.load(wav_bytes, sr = None, mono=False)
    if sr is not None:
        audio = librosa.resample( orig_audio, orig_sr = orig_sr, target_sr = sr )
    else:
        sr = orig_sr
        audio = orig_audio.copy()
    
    return audio, sr, orig_audio, orig_sr

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

def get_spectrogram( audio, sr, start_time, hop_length, 
                     num_spec_columns, 
                     min_frequency = None, max_frequency = None, n_bins = 256,
                     spec_cal_method = None,
                     n_fft = None,
                     bins_per_octave = None,
                     brightness=1.0, contrast=1.0,
                     color_map = "inferno"
                   ):
    if color_map not in [ 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray' ]:
        color_map = 'inferno'

    if spec_cal_method is None:
        spec_cal_method = "log-mel"
                
    
    hop_length = int(hop_length)
    num_samples = hop_length * num_spec_columns

    if spec_cal_method == "log-mel":
        spec_cal = SpecCalLogMel( sr = sr, hop_length = hop_length, 
                                  min_frequency = min_frequency, max_frequency = max_frequency,
                                  n_bins = n_bins, n_fft = n_fft, color_map = color_map )
    elif spec_cal_method == "constant-q":
        spec_cal = SpecCalConstantQ( sr = sr, hop_length = hop_length, 
                                  min_frequency = min_frequency, max_frequency = max_frequency,
                                  n_bins = n_bins, bins_per_octave = bins_per_octave,  color_map = color_map )
    elif spec_cal_method == "dummy":
        spec_cal = SpecCalDummy( sr = sr, hop_length = hop_length, 
                                  min_frequency = min_frequency, max_frequency = max_frequency,
                                  n_bins = n_bins, n_fft = n_fft,  color_map = color_map )
    else:
        assert False, "Unsupported spectrogram computation method!"
    
    start_time = max( start_time, 0.0 )    
    audio_clip = audio[ int( start_time * sr ): int( start_time * sr ) + num_samples ]
    audio_clip = np.concatenate( [ audio_clip, np.zeros( num_samples - len(audio_clip) ) ], axis = 0 )
    audio_clip = audio_clip.astype(np.float32)
        
    log_mel_spec = spec_cal( audio_clip, brightness = brightness, contrast = contrast )
    
    ## resize the log_mel_spec
    log_mel_spec = resize( log_mel_spec, ( n_bins, num_spec_columns, 3 ) )

    config = {
        "spec_cal_method":spec_cal_method,
        "n_fft": spec_cal.n_fft if spec_cal_method in [ "log-mel" ] else n_fft,
        "bins_per_octave": spec_cal.bins_per_octave if spec_cal_method in [ "constant-q" ] else bins_per_octave,
        "hop_length": hop_length,
        "num_spec_columns":num_spec_columns,
        "sampling_rate":sr,
        "min_frequency":spec_cal.min_frequency,
        "max_frequency":spec_cal.max_frequency
    }

    return log_mel_spec, spec_cal.freqs, config

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

def normalize_audio( audio ):
    if len(audio) == 0:
        return audio
    std = np.std(audio)
    avg = np.mean(audio)
    normalized_audio = (audio - avg) / (std + 1e-6)
    return normalized_audio


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
    
def register_new_audio( audio, sr, orig_audio, orig_sr, audio_id  ): 
    ## "audio" is only used when displaying the spectrogram
    ## "orig_audio" is used for get wavform, and run segmentation
    global audio_dict
    audio_dict[audio_id] = {
        "audio":audio,
        "sr":sr,
        "orig_audio":orig_audio,
        "orig_sr":orig_sr,
        "percentile_up":np.percentile(orig_audio, 99.99),
        "percentile_down":np.percentile(orig_audio, 0.01),
        "timestamp":datetime.now()
    }   

def release_idle_audios( audio_dict, idle_hours ):
    print("Daemon thread for releasing idle audios is running ...")
    while True:
        current_time = datetime.now()
        audio_ids_to_release = []
        for audio_id in audio_dict:
            if current_time - audio_dict[audio_id]["timestamp"] > timedelta(hours=idle_hours):
                audio_ids_to_release.append( audio_id )
        for audio_id in audio_ids_to_release:
            try:
                del audio_dict[audio_id]
            except:
                pass
            print( "Audio %s has been deleted after being idle over %f hours"%(audio_id, idle_hours) )
        time.sleep(60 * 60)  # Check every hour
        
def parse_clustername( cluster, cluster_separator ):
    res = re.findall( r"^(.*?)%s(.*?)%s(.*?)$"%( cluster_separator, cluster_separator ), cluster )
    if len(res) == 1:
        parsed_res = {
            "species":res[0][0],
            "individual":res[0][1],
            "clustername":res[0][2],
        }
    else:
        parsed_res = {
            "species":"Unknown",
            "individual":"Unknown",
            "clustername": cluster # "Unknown"
        }
    return parsed_res

"""Example timestamps for plotting
timestamps = ['2024-11-20 16:42:14.241', 
              '2024-11-20 16:42:14.880', 
              '2024-11-20 16:43:15.391', 
              '2024-11-20 16:43:16.546', 
              '2024-11-20 16:50:17.535', 
              '2024-11-20 16:50:19.173', 
              '2024-11-20 16:50:28.419', 
              '2024-11-20 16:51:30.223', 
              '2024-11-20 16:52:31.626', 
              '2024-11-20 16:52:32.200', 
              '2024-11-20 16:58:34.332', 
              '2024-11-20 16:59:34.922', 
              '2024-11-20 16:59:39.400', 
              '2024-11-20 16:59:40.857', 
              '2024-11-20 16:59:42.629']

"""
def get_estimated_annotation_time(timestamps, idle_thres = 300 ):
    timestamps = [ datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in timestamps]
    timestamps = sorted(timestamps)
    if len(timestamps) <2:
        return 0
    
    time_delta_list = [0]
    for pos in range(len(timestamps) - 1):
        time_delta_list.append( (timestamps[pos + 1] - timestamps[pos]).total_seconds() )
    
    total_time = 0
    for dt in time_delta_list:
        if dt <= idle_thres:
            total_time += dt
    
    return total_time

def plot_timestamp_vs_annotation_action( timestamps ):
    timestamps = [ datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in timestamps]
    timestamps = sorted(timestamps)
    if len(timestamps) <2:
        return 
    
    timestamps = [ (t - timestamps[0]).total_seconds() for t in timestamps ]
    plt.plot( timestamps)
    plt.xticks(range(len(timestamps))) 
    plt.xlabel("Annotation actions")
    plt.ylabel("Timestamps (s)")
    plt.show()


@app.route("/get-status", methods=['GET'])
def get_status():
    return jsonify({"response":"success"}), 201
    
@app.route("/upload", methods=['POST'])
def upload():
    global audio_dict, n_bins
    
    newAudioFile = request.files['newAudioFile']
    spec_cal_method = request.form.get('spec_cal_method', type=str, default=None)
    n_fft = request.form.get('n_fft', type=int, default=None)
    bins_per_octave = request.form.get('bins_per_octave', type=int, default=None)
    hop_length = request.form.get('hop_length', type=int, default=None)
    num_spec_columns = request.form.get('num_spec_columns', type=int, default=None)
    sr = request.form.get('sampling_rate', type=int, default=None)
    min_frequency = request.form.get('min_frequency', type=int, default=None)
    max_frequency = request.form.get('max_frequency', type=int, default=None)
    color_map = request.form.get('color_map', type=str, default="inferno")
    

    if num_spec_columns is None:
        num_spec_columns = 1000

    print("File received!", datetime.now())

    ## for certain reason after Flask transfer b"\r\n" can be prefixed to the data, so we need to remove them, otherwise the librosa load will trigger error
    orig_audio_multi_channels, orig_sr = librosa.load( io.BytesIO(newAudioFile.read().lstrip()), sr = None, mono = False ) 
    if sr is not None:
        audio_multi_channels = librosa.resample( orig_audio_multi_channels, orig_sr = orig_sr, target_sr = sr )
    else:
        audio_multi_channels = orig_audio_multi_channels.copy()
        sr = orig_sr
    
    if max_frequency is None or max_frequency <= 0 :
        max_frequency = sr // 2
    else:
        max_frequency = min( max_frequency, sr//2 )
        
    if len( audio_multi_channels.shape ) == 1:
        audio_multi_channels = audio_multi_channels[ np.newaxis,: ]
        orig_audio_multi_channels = orig_audio_multi_channels[ np.newaxis,: ]

    ## multiple channels of the same audio should have the same length
    if hop_length is None:
        hop_length = audio_multi_channels.shape[1] // num_spec_columns
    
    spec_all_channels = []
    for pos in range( audio_multi_channels.shape[0] ):
        audio = audio_multi_channels[pos]
        orig_audio = orig_audio_multi_channels[pos]

        audio_id = str( uuid4() )
        
        whole_audio_spec, freqs, config  = get_spectrogram( audio, sr, 0, hop_length, 
                                                          num_spec_columns = num_spec_columns, 
                                                          min_frequency = min_frequency, 
                                                          max_frequency = max_frequency,
                                                          n_bins = n_bins,
                                                          spec_cal_method = spec_cal_method,
                                                          n_fft = n_fft,
                                                          bins_per_octave = bins_per_octave,
                                                          color_map = color_map
                                                        )
        
        register_new_audio( audio, sr, orig_audio, orig_sr, audio_id )
        
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

    results = {
        "configurations": config,
        "channels": spec_all_channels
    }
    return jsonify(results), 201

@app.route("/upload-by-url", methods=['POST'])
def upload_by_url():
    global audio_dict, num_spec_columns, n_bins
    request_info = request.json
    audio_url = request_info['audio_url']
    spec_cal_method = request_info.get('spec_cal_method', None)
    n_fft = request_info.get('n_fft', None)
    bins_per_octave = request_info.get('bins_per_octave', None)
    hop_length = request_info.get('hop_length', None)
    num_spec_columns = request_info.get('num_spec_columns', None)
    sr = request_info.get('sampling_rate', None)
    min_frequency = request_info.get('min_frequency', None)
    max_frequency = request_info.get('max_frequency', None)
    color_map = request_info.get('color_map', "inferno")

    if num_spec_columns is None:
        num_spec_columns = 1000

    audio_multi_channels, sr, orig_audio_multi_channels, orig_sr = load_audio_from_url(audio_url, sr)
    if len( audio_multi_channels.shape ) == 1:
        audio_multi_channels = audio_multi_channels[ np.newaxis,: ]
        orig_audio_multi_channels = orig_audio_multi_channels[ np.newaxis,: ]

    if max_frequency is None:
        max_frequency = sr // 2
    else:
        max_frequency = min( max_frequency, sr//2 )

    ## multiple channels of the same audio should have the same length
    if hop_length is None:
        hop_length = audio_multi_channels.shape[1] // num_spec_columns

    spec_all_channels = []
    for pos in range( audio_multi_channels.shape[0] ):
        audio = audio_multi_channels[pos]
        orig_audio = orig_audio_multi_channels[pos]
    
        audio_id = str( uuid4() )
        
        whole_audio_spec, freqs, config = get_spectrogram( 
                                                          audio, sr, 0, hop_length, 
                                                          num_spec_columns = num_spec_columns, 
                                                          min_frequency = min_frequency, 
                                                          max_frequency = max_frequency,
                                                          n_bins = n_bins,
                                                          spec_cal_method = spec_cal_method,
                                                          n_fft = n_fft,
                                                          bins_per_octave = bins_per_octave,
                                                          color_map = color_map
                                                        )
        register_new_audio( audio, sr, orig_audio, orig_sr, audio_id )
        
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

    results = {
        "configurations": config,
        "channels": spec_all_channels
    }
    return jsonify(results), 201
    
@app.route("/get-audio-clip-spec", methods=['POST'])
def get_audio_clip_spec():
    global audio_dict, num_spec_columns, n_bins
    
    request_info = request.json

    audio_id = request_info["audio_id"]
    start_time = request_info["start_time"]

    spec_cal_method = request_info['spec_cal_method']
    n_fft = request_info['n_fft']
    bins_per_octave = request_info['bins_per_octave']
    hop_length = request_info["hop_length"]
    num_spec_columns = request_info['num_spec_columns']
    sr = request_info['sampling_rate']
    min_frequency = request_info['min_frequency']
    max_frequency = request_info['max_frequency']
    brightness = request_info.get( "brightness", 1.0) 
    contrast = request_info.get( "contrast", 1.0) 
    color_map = request_info.get('color_map', "inferno")
    
    audio = audio_dict[audio_id]["audio"]  

    ## update the timestamp of the audio_id
    audio_dict[audio_id]["timestamp"] = datetime.now()
        
    if sr is None:
        sr = audio_dict[audio_id]["sr"]
    else:
        ## check if the given sr is equal to the audio_dict[audio_id]["sr"]
        if sr != audio_dict[audio_id]["sr"]:
            ## resampling the audio array with the given sr
            audio = librosa.resample(audio_dict[audio_id]["orig_audio"], 
                                     orig_sr=audio_dict[audio_id]["orig_sr"], 
                                     target_sr=sr)
            audio_dict[audio_id]["audio"] = audio
            audio_dict[audio_id]["sr"] = sr
    
    if max_frequency is None:
        max_frequency = sr // 2
    else:
        max_frequency = min( max_frequency, sr//2 )
    
    audio_clip_spec, freqs, config = get_spectrogram( 
                                                          audio, sr, start_time, hop_length, 
                                                          num_spec_columns = num_spec_columns, 
                                                          min_frequency = min_frequency, 
                                                          max_frequency = max_frequency,
                                                          n_bins = n_bins,
                                                          spec_cal_method = spec_cal_method,
                                                          n_fft = n_fft,
                                                          bins_per_octave = bins_per_octave,
                                                          brightness = brightness,
                                                          contrast = contrast,
                                                          color_map = color_map
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
            "freqs": freqs.tolist(),
            "configurations": config
           }

@app.route("/get-audio-clip-wav", methods=['POST'])
def get_audio_clip_wav():
    global audio_dict
    
    request_info = request.json
    audio_id = request_info["audio_id"]

    start_time = request_info["start_time"]
    clip_duration = request_info["clip_duration"]
    
    ## update the timestamp of the audio_id
    audio_dict[audio_id]["timestamp"] = datetime.now()
    
    audio = audio_dict[audio_id]["orig_audio"]
    sr = audio_dict[audio_id]["orig_sr"]
    
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
    
    ## update the timestamp of the audio_id
    audio_dict[audio_id]["timestamp"] = datetime.now()
    
    audio = audio_dict[audio_id]["orig_audio"]
    sr = audio_dict[audio_id]["orig_sr"]
    percentile_up = audio_dict[audio_id]["percentile_up"]
    percentile_down = audio_dict[audio_id]["percentile_down"]
    
    audio_clip = audio[ int( start_time * sr ):int( (start_time + clip_duration) * sr ) ]
    audio_clip = resample_audio( audio_clip, target_length )

    # audio_clip = np.clip( audio_clip, a_min = percentile_down, a_max =  percentile_up )
    audio_clip = normalize_audio(audio_clip)
    return jsonify({"wav_array":audio_clip.tolist()}), 201

@app.route("/list-models-available-for-finetuning", methods=['POST'])
def list_models_available_for_finetuning():
    global args    
    res = requests.post( args.segmentation_service_address + "/list-models-available-for-finetuning" ).json()
    return jsonify(res), 201

@app.route("/list-models-available-for-inference", methods=['POST'])
def list_models_available_for_inference():
    global args    
    res = requests.post( args.segmentation_service_address + "/list-models-available-for-inference" ).json()
    return jsonify(res), 201

@app.route("/list-models-training-in-progress", methods=['POST'])
def list_models_being_trained():
    global args    
    res = requests.post( args.segmentation_service_address + "/list-models-training-in-progress" ).json()
    return jsonify(res), 201

@app.route("/get-labels", methods=['POST'])
def get_labels():
    global audio_dict, cluster_separator, args
    
    request_info = request.json
    audio_id = request_info["audio_id"]
    annotated_areas = request_info.get("annotated_areas",[])
    human_labels = request_info.get("human_labels", [])
    model_name = request_info.get("model_name", "whisperseg-base")
    min_frequency = request_info.get( "min_frequency", None )
    token = request_info.get("token", "")

    """
    Perform authorization HERE:
    """
    if token not in ["","123"]:
        return jsonify( {"error":"unauthorized"} ), 403
        
    
    ## update the timestamp of the audio_id
    audio_dict[audio_id]["timestamp"] = datetime.now()
    audio = audio_dict[audio_id]["orig_audio"]
    sr = audio_dict[audio_id]["orig_sr"]
    
    prediction = segment_audio( args.segmentation_service_address, model_name, audio, sr, min_frequency = min_frequency, spec_time_step = None )
    for item in prediction:
        item.update( parse_clustername( item["cluster"], cluster_separator ) )
        del item["cluster"]
        
    human_labels_in_annotated_areas = []
    for item in human_labels:
        is_in_annotated_area = False
        for area in annotated_areas:
            if item["onset"] < area["offset"] and item["offset"] > area["onset"]:
                is_in_annotated_area = True
                break
        if is_in_annotated_area:
            human_labels_in_annotated_areas.append({
                "onset":item["onset"],
                "offset":item["offset"],
                "species":item["species"],
                "individual":item["individual"],
                "clustername":item["clustername"]
            })
            
    prediction_not_in_annotated_areas = []
    for item in prediction:
        is_in_annotated_area = False
        for area in annotated_areas:
            if item["onset"] < area["offset"] and item["offset"] > area["onset"]:
                is_in_annotated_area = True
                break
        if not is_in_annotated_area:
            prediction_not_in_annotated_areas.append(item)
            
    final_prediction = human_labels_in_annotated_areas + prediction_not_in_annotated_areas
    final_prediction.sort( key = lambda x:x["onset"] )
        
    return jsonify({"labels":final_prediction}), 201

@app.route("/finetune-whisperseg", methods=['POST'])
def finetune_whisperseg():
    global audio_dict, cluster_separator, args
    request_info = request.json
    audio_id = request_info["audio_id"]
    annotated_areas = request_info["annotated_areas"]
    human_labels = request_info["human_labels"]
    new_model_name = request_info["new_model_name"]
    initial_model_name = request_info["initial_model_name"]
    min_frequency = request_info.get( "min_frequency", None )
    token = request_info.get("token", "")

    """
    Perform authorization HERE:
    """
    if token not in ["","123"]:
        return jsonify( {"error":"unauthorized"} ), 403
    
    
    ## update the timestamp of the audio_id
    audio_dict[audio_id]["timestamp"] = datetime.now()
    audio = audio_dict[audio_id]["orig_audio"]
    sr = audio_dict[audio_id]["orig_sr"]
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', compression=zipfile.ZIP_STORED) as zipf:
        file_count = 0
        for count, area in enumerate(annotated_areas):
            area_onset = area["onset"]
            area_offset = area["offset"]
            audio_clip = audio[ int( area_onset * sr ):int( area_offset * sr ) ] 
            if len(audio_clip) == 0:
                continue
            ## compute the actual area_offset
            area_offset = area_onset + len( audio_clip ) / sr
            
            human_labels_with_in_area = []
            for item in human_labels:
                if item["onset"] < area_offset and item["offset"] > area_onset and item["offset"] > item["onset"]:
                    human_labels_with_in_area.append(
                        {
                            "onset":max(0, item["onset"] - area_onset),
                            "offset":min( area_offset - area_onset, item["offset"] - area_onset ),
                            "cluster": cluster_separator.join( [item["species"], item["individual"], item["clustername"]] ) 
                        }
                    )
            human_labels_with_in_area.sort( key = lambda x:x["onset"] )
            label_clip = {
                "onset":[ item["onset"] for item in human_labels_with_in_area],
                "offset":[ item["offset"] for item in human_labels_with_in_area],
                "cluster":[ item["cluster"] for item in human_labels_with_in_area]
            }
            if isinstance( min_frequency, int ):
                label_clip["min_frequency"] = min_frequency
            
            audio_fname = "%s_%d.wav"%( audio_id, count )
            label_fname = "%s_%d.json"%( audio_id, count )
            buffer = io.BytesIO()
            sf.write(buffer, audio_clip, samplerate=sr, format='WAV')
            buffer.seek(0)
            zipf.writestr(audio_fname, buffer.read())
            buffer.close()
            
            buffer = io.BytesIO()
            buffer.write(json.dumps(label_clip).encode('utf-8'))
            buffer.seek(0)
            zipf.writestr(label_fname, buffer.read())
            buffer.close()
            
            file_count += 1
            
    ## This is necessary!
    memory_file.seek(0)
        
    if file_count == 0:
        return jsonify({"error":"No valid training data specified. Re-check the annotated areas."}), 400
    
    response, status_code = submit_training_request( args.segmentation_service_address, new_model_name, initial_model_name, memory_file, num_epochs = 3 )
    status_code = 201 if status_code != 400 else 400
    
    return jsonify( response ), status_code

@app.route("/get-metadata/", methods=['POST'])
def get_metadata():
    global args
    try:
        request_info = request.json 
        hash_id = request_info["hash_id"]
        user_profile = request_info.get( "user_profile", False )
        if user_profile:
            res = requests.get( args.vocallbase_service_address + "/get-audio-user-config-from-hash-id/" + hash_id ).json()
        else:
            res = requests.get( args.vocallbase_service_address + "/get-audio-dataset-config-from-hash-id/" + hash_id ).json()
        res = [ res ]
    except:
        res = []
    
    return jsonify(res), 201


@app.route("/get-annotations/<hash_id>", methods=['GET'])
def get_annotations(hash_id):
    global args
    try:
        res = requests.get(f"{args.vocallbase_service_address}/annotations/{hash_id}").json()
    except:
        res = []
    
    return jsonify(res), 201

@app.route("/post-annotations/<hash_id>", methods=['POST'])
def post_annotations(hash_id):
    global args

    res = requests.post(
        args.vocallbase_service_address + f"/annotations/{hash_id}/",
        data = json.dumps(request.json),
        headers = { "Content-Type":"application/json",
                    "accept":"application/json"
                  }
    )

    status_code = res.status_code
    if status_code == 200 or status_code == 201:
        status_code = 201
    else:
        status_code = 400
    try:
        response = res.json()
    except:
        response = {"Warning:":"No response in posting annotation"}
        
    return jsonify(response), status_code


@app.route("/release-audio-given-ids", methods=['POST'])
def release_audio_given_ids():
    global audio_dict
    request_info = request.json
    audio_id_list = request_info["audio_id_list"]
    for audio_id in audio_id_list:
        try:
            del audio_dict[str(audio_id)]
            print( "%s is deleted from audio_dict"%( str(audio_id) ))
        except:
            print("Warning: %s in not in audio_dict"%( str(audio_id) ))
            pass
    
    return jsonify({"status":"success"}), 201


@app.route("/estimate-annotation-time", methods=['POST'])
def estimate_annotation_time():
    request_info = request.json
    timestamps = request_info["timestamps"]
    idle_thres = request_info.get("idle_thres", 300)  ## setting the default idle threshold as 5 min
    anno_time = get_estimated_annotation_time(timestamps, idle_thres = idle_thres )
    return jsonify({"response":anno_time}), 201


@app.route("/annotations/annotation-time/<hash_id>", methods=['POST'])
def post_annotation_time(hash_id):
    global args

    print("submitted data:", request.json)

    res = requests.post(
        args.vocallbase_service_address + f"/annotations/annotation-time/{hash_id}/",
        data = json.dumps(request.json),
        headers = { "Content-Type":"application/json",
                    "accept":"application/json"
                  }
    )

    status_code = res.status_code
    if status_code == 200 or status_code == 201:
        status_code = 201
    else:
        status_code = 400
    try:
        response = res.json()
    except:
        response = {"Warning:":"No response from posting annotation time"}
        
    return jsonify(response), status_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-flask_port", help="The port of the flask app.", default=8050, type=int)
    parser.add_argument("-segmentation_service_address", help="The address to the WhisperSeg segmentation API.")
    parser.add_argument("-vocallbase_service_address", help="The address to the dataplatform evolving language.")
    args = parser.parse_args()
    
    audio_dict = {}
    num_spec_columns = 1000
    n_bins = 300
    idle_hours = 48
    cluster_separator = "<==SEPARATOR==>"
    
    thread = threading.Thread(target=release_idle_audios, args=(audio_dict, idle_hours))
    thread.daemon = True
    thread.start()
    
    print( "Waiting for requests..." )

    app.run( host='0.0.0.0', port=args.flask_port, debug = True )
