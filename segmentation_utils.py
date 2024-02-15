import requests,json,base64
import pandas as pd
import numpy as np
import io
import soundfile as sf

## define a function for segmentation
def call_segment_service( service_address, 
                          audio,
                          sr,
                          min_frequency = None,
                          spec_time_step = None,
                          min_segment_length = None,
                          eps = None,
                          num_trials = 1
                        ):

    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=sr, format='WAV')
    # Rewind the buffer to the beginning so we can read its contents
    buffer.seek(0)
    audio_file_base64_string = base64.b64encode(buffer.read()).decode('ASCII')
    
    ### Empirically determine the min_frequency, spec_time_step, min_segment_length, eps
    if min_frequency is None:
        min_frequency = 0
    if spec_time_step is None:
        spec_time_step = 0.0025 if sr < 100000 else 0.0005
    if min_segment_length is None:
        min_segment_length = 0.01
    if eps is None:
        eps = 0.02
    
    response = requests.post( service_address,
                              data = json.dumps( {
                                  "audio_file_base64_string":audio_file_base64_string,
                                  "sr":sr,
                                  "min_frequency":min_frequency,
                                  "spec_time_step":spec_time_step,
                                  "min_segment_length":min_segment_length,
                                  "eps":eps,
                                  "num_trials":num_trials,
                              } ),
                              headers = {"Content-Type": "application/json"}
                            )
    prediction = response.json()
    final_prediction = [ {
        "onset":float(onset),
        "offset":float(offset),
        "clustername":None
        } for onset, offset, cluster in zip( prediction["onset"], prediction["offset"], prediction["cluster"])]
    
    return final_prediction