import requests,json,base64
import pandas as pd
import numpy as np
import io
import soundfile as sf


def segment_audio( service_address, model_name, audio, sr, min_frequency = None, spec_time_step = None, channel_id = 0 ):
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=sr, format='WAV')
    # Rewind the buffer to the beginning so we can read its contents
    buffer.seek(0)
    response = requests.post( service_address + "/segment", files = { "audio_file": buffer.read() },
                                data = { "model_name":model_name,
                                            "min_frequency":min_frequency,
                                            "spec_time_step":spec_time_step,
                                            "channel_id": channel_id
                                          })
    prediction = response.json()
    final_prediction = [ {
        "onset":float(onset),
        "offset":float(offset),
        "cluster":cluster
        } for onset, offset, cluster in zip( prediction["onset"], prediction["offset"], prediction["cluster"])]
    
    return final_prediction

def submit_training_request( service_address, model_name, inital_model_name, memory_file, num_epochs = 3 ):
    files = { "zip":memory_file }
    response = requests.post( service_address + "/submit-training-request", files=files, data = { "model_name":model_name,
                                                        "inital_model_name":inital_model_name,
                                                        "num_epochs":num_epochs
                                                      })

    return response.json(), response.status_code

