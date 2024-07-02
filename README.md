# Installation of environment
## Method 1
Install via pip:
Create an Anaconda environment
```bash
conda create -n audio_backend python=3.10
```
Activate the environment
```bash
conda activate audio_backend
```
Install dependices via pip
```bash
pip install -r requirements.txt
```
Install FFMPEG (needed in order to process MP3 audios correctly)
```bash
conda install conda-forge::ffmpeg
```
## Method 2
Install via environment.yml file
```bash
conda env create -f environment.yml
```

# Start service
Activate the anaconda environment "audio_backend":
```bash
conda activate audio_backend
```
In the main folder, run
```bash
python backend.py
```

# API Documentation 

```python
import requests, json
import base64
from IPython.display import Audio
import io
from PIL import Image
import base64
```


```python
def bytes_to_base64_string(f_bytes):
    return base64.b64encode(f_bytes).decode('ASCII')

def base64_string_to_bytes(base64_string):
    return base64.b64decode(base64_string)
```

## Upload

### Upload from file


```python
with open("example_audios/BP_2021-10-23_09-08-47_049985_0540000_daq1.wav", "rb") as f:
    file_data = f.read()
```


```python
results = requests.post(
    "http://localhost:8050/upload",
    files = {"newAudioFile":file_data},
    data = { 
             "hop_length": None,
             "num_spec_columns": None,
             "sampling_rate": None,
             "spec_cal_method": None,
             "n_fft": None,
             "bins_per_octave": None,
             "min_frequency": None,
             "max_frequency": None
           }
).json()
audio_id = results["channels"][0]["audio_id"]
```


```python
results["channels"][0]["audio_duration"]
```




    419.4304375




```python
results["configurations"]
```




    {'bins_per_octave': None,
     'hop_length': 6710,
     'max_frequency': 8000,
     'min_frequency': 0,
     'n_fft': 512,
     'num_spec_columns': 1000,
     'sampling_rate': 16000,
     'spec_cal_method': 'log-mel'}



### Upload from URL


```python
import requests, json
results = requests.post( 
    "http://localhost:8050/upload-by-url",
    data = json.dumps( {
             "audio_url":"https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav" ,
             "spec_cal_method": "constant-q",
             "n_fft": None,
             "bins_per_octave": None,
             "hop_length": None,
             "num_spec_columns": None,
             "sampling_rate": None,
             "min_frequency": -100,
             "max_frequency": None
    }),
    headers = {"content-type":"application/json"}
).json()
```

## Get spectrogram


```python
import requests, json
results = requests.post( 
    "http://localhost:8050/get-audio-clip-spec",
    data = json.dumps( {
             "audio_id":audio_id ,
             "start_time": 1.2,
             "spec_cal_method": None,
             "n_fft": None,
             "bins_per_octave": None,
             "hop_length": 160,
             "num_spec_columns": 1000,
             "sampling_rate": None,
             "min_frequency": 0,
             "max_frequency": None
    }),
    headers = {"content-type":"application/json"}
).json()
```


```python
results.keys()
```




    dict_keys(['configurations', 'freqs', 'spec'])




```python
results["configurations"]
```




    {'bins_per_octave': None,
     'hop_length': 160,
     'max_frequency': 8000,
     'min_frequency': 0,
     'n_fft': 512,
     'num_spec_columns': 1000,
     'sampling_rate': 16000,
     'spec_cal_method': 'log-mel'}



## Get Audio Clip


```python
response = requests.post( 
    "http://localhost:8050/get-audio-clip-wav",
    data = json.dumps( {
        "audio_id":audio_id,
        "start_time":0,
        "clip_duration":100,
    }),
    headers = {"content-type":"application/json"}
).json()
Audio(base64.b64decode(response["wav"]))
```

## Post Labels


```python
import requests, json

res = requests.post(
    'http://localhost:8050/post-annotations',
    data = json.dumps(
        {  
            "annotations": [
                {
                    "onset": 0,
                    "offset": 0,
                    "species": "SPECIES_NAME_HERE",
                    "individual": "INDIVIDUAL_NAME_HERE",
                    "filename": "FILENAME_HERE",
                    "annotation_instance": "ANNOTATION_INSTANCE_HERE"
                },
                ##  more annotations goes here
            ]
        
        }
    ),
    headers = { "Content-Type":"application/json",
                "accept":"application/json"
              }
).json()
res
```




    {'message': 'Annotations inserted successfully.'}



## Delete Audio Ids


```python
import requests, json

res = requests.post(
    'http://localhost:8050/release-audio-given-ids',
    data = json.dumps({ "audio_id_list": [ '152ca390-7ca7-48a0-b0c4-aa9617639753',
                                           'ddjjiu3m-huue-efrw-frff-bshah8773ksu',
                                         ] }),
    headers = { "Content-Type":"application/json",
                "accept":"application/json"
              }
).json()
res
```




    {'status': 'success'}



## List Available Models

### List Models Available for Finetuning


```python
response = requests.post( 
    "http://localhost:8050/list-models-available-for-finetuning",
    headers = {"content-type":"application/json"}
).json()
response
```




    {'response': [{'eta': '--:--:--',
       'model_name': 'whisperseg-base',
       'status': 'ready'},
      {'eta': '--:--:--', 'model_name': 'whisperseg-large', 'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'r3428-99dph-whisperseg_base',
       'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'r3428-99dph-whisperseg-base-v2.0',
       'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'r3428-99dph-whisperseg-large',
       'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'new-whisperseg-bengalese-finch',
       'status': 'ready'}]}



### List Models Available for Inference


```python
response = requests.post( 
    "http://localhost:8050/list-models-available-for-inference",
    headers = {"content-type":"application/json"}
).json()
response
```




    {'response': [{'eta': '--:--:--',
       'model_name': 'whisperseg-base',
       'status': 'ready'},
      {'eta': '--:--:--', 'model_name': 'whisperseg-large', 'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'r3428-99dph-whisperseg_base',
       'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'r3428-99dph-whisperseg-base-v2.0',
       'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'r3428-99dph-whisperseg-large',
       'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'new-whisperseg-bengalese-finch',
       'status': 'ready'}]}



### List Models Being Trained


```python
response = requests.post( 
    "http://localhost:8050/list-models-training-in-progress",
    headers = {"content-type":"application/json"}
).json()
response
```




    {'response': []}



## Get WhisperSeg Segmentation


```python
response = requests.post( 
    "http://localhost:8050/get-labels",
    data = json.dumps( {
        "audio_id":audio_id,
        "model_name":"whisperseg-large",
        "min_frequency": 0
    }),
    headers = {"content-type":"application/json"}
).json()
```


```python
response["labels"][:5]
```




    [{'clustername': 'Unknown',
      'individual': 'Unknown',
      'offset': 14.595,
      'onset': 14.497,
      'species': 'Unknown'},
     {'clustername': 'Unknown',
      'individual': 'Unknown',
      'offset': 15.108,
      'onset': 15.008,
      'species': 'Unknown'},
     {'clustername': 'Unknown',
      'individual': 'Unknown',
      'offset': 15.695,
      'onset': 15.572,
      'species': 'Unknown'},
     {'clustername': 'Unknown',
      'individual': 'Unknown',
      'offset': 16.772,
      'onset': 16.715,
      'species': 'Unknown'},
     {'clustername': 'Unknown',
      'individual': 'Unknown',
      'offset': 17.028,
      'onset': 16.983,
      'species': 'Unknown'}]



## Human-in-the-loop Training WhisperSeg Pipeline


```python
import requests, json
import pandas as pd
```

### Upload file


```python
with open("example_audios/human-in-the-loop-training-example/audio.wav", "rb") as f:
    file_data = f.read()
results = requests.post(
    "http://localhost:8050/upload",
    files = {"newAudioFile":file_data},
    data = { 
             "hop_length": None,
             "num_spec_columns": None,
             "sampling_rate": None,
             "spec_cal_method": None,
             "n_fft": None,
             "bins_per_octave": None,
             "min_frequency": None,
             "max_frequency": None
           }
).json()
audio_id = results["channels"][0]["audio_id"]
audio_id
```




    '41525b82-f471-4699-85b0-1cfe44ed828b'



### Human Annotation 
Suppose human annotator annotated the first 15 seconds


```python
df = pd.read_csv( "example_audios/human-in-the-loop-training-example/annotations.csv" )
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>onset</th>
      <th>offset</th>
      <th>species</th>
      <th>individual</th>
      <th>clustername</th>
      <th>filename</th>
      <th>channelIndex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.121816</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>U</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.178039</td>
      <td>0.271743</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>U</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.290484</td>
      <td>0.393559</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>U</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.440411</td>
      <td>0.599709</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>A</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.599709</td>
      <td>0.862082</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>B</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Check the available model for finetuning and select one


```python
response = requests.post( 
    "http://localhost:8050/list-models-available-for-finetuning",
    headers = {"content-type":"application/json"}
).json()
response
```




    {'response': [{'eta': '--:--:--',
       'model_name': 'whisperseg-base',
       'status': 'ready'},
      {'eta': '--:--:--', 'model_name': 'whisperseg-large', 'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'r3428-99dph-whisperseg_base',
       'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'r3428-99dph-whisperseg-base-v2.0',
       'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'r3428-99dph-whisperseg-large',
       'status': 'ready'},
      {'eta': '--:--:--',
       'model_name': 'new-whisperseg-bengalese-finch',
       'status': 'ready'},
      {'eta': '--:--:--', 'model_name': 'whisperseg-new-v1.0', 'status': 'ready'}]}



suppose the user selects the model "whisperseg-base"

### Give a new name to the finetuned model
For this the user can see the full list of existing models, and the frontend will check if the given new_model_name is unique

The list of existing model names can be obtained by the API calls below:


```python
all_models = requests.post(  "http://localhost:8050/list-models-available-for-finetuning" ).json()["response"] + \
requests.post(  "http://localhost:8050/list-models-available-for-inference" ).json()["response"] + \
requests.post(  "http://localhost:8050/list-models-training-in-progress" ).json()["response"] 
all_models
```




    [{'eta': '--:--:--', 'model_name': 'whisperseg-base', 'status': 'ready'},
     {'eta': '--:--:--', 'model_name': 'whisperseg-large', 'status': 'ready'},
     {'eta': '--:--:--',
      'model_name': 'r3428-99dph-whisperseg_base',
      'status': 'ready'},
     {'eta': '--:--:--',
      'model_name': 'r3428-99dph-whisperseg-base-v2.0',
      'status': 'ready'},
     {'eta': '--:--:--',
      'model_name': 'r3428-99dph-whisperseg-large',
      'status': 'ready'},
     {'eta': '--:--:--',
      'model_name': 'new-whisperseg-bengalese-finch',
      'status': 'ready'},
     {'eta': '--:--:--', 'model_name': 'whisperseg-new-v1.0', 'status': 'ready'},
     {'eta': '--:--:--', 'model_name': 'whisperseg-base', 'status': 'ready'},
     {'eta': '--:--:--', 'model_name': 'whisperseg-large', 'status': 'ready'},
     {'eta': '--:--:--',
      'model_name': 'r3428-99dph-whisperseg_base',
      'status': 'ready'},
     {'eta': '--:--:--',
      'model_name': 'r3428-99dph-whisperseg-base-v2.0',
      'status': 'ready'},
     {'eta': '--:--:--',
      'model_name': 'r3428-99dph-whisperseg-large',
      'status': 'ready'},
     {'eta': '--:--:--',
      'model_name': 'new-whisperseg-bengalese-finch',
      'status': 'ready'},
     {'eta': '--:--:--', 'model_name': 'whisperseg-new-v1.0', 'status': 'ready'}]



### Start submit training request


```python
# audio_id
annotated_areas = [ { "onset":0, "offset":45 } ]  ## the first 15 seconds are annotated
human_labels = [dict(df.iloc[idx]) for idx in range(len(df)) ]  ## get the human_labels
for item in human_labels: ## to make it JSON serializable
    item["onset"] = float(item["onset"])
    item["offset"] = float(item["offset"])  
    item["channelIndex"] = int(item["channelIndex"])

new_model_name = "whisperseg-base-debug-v1.0"
inital_model_name = "whisperseg-base"
min_frequency = 0
```


```python
response = requests.post( 
    "http://localhost:8050/finetune-whisperseg",
    data = json.dumps({
        "audio_id":audio_id,
        "annotated_areas":annotated_areas,
        "human_labels":human_labels,
        "new_model_name":new_model_name,
        "inital_model_name":inital_model_name,
        "min_frequency":min_frequency
    }),
    headers = {"content-type":"application/json"}
).json()

response
```




    {'message': 'Training'}



Let's see the status of model being trained


```python
requests.post(  "http://localhost:8050/list-models-training-in-progress" ).json()["response"] 
```




    []



### Use the finetuned model to segment the rest of the audios


```python
response = requests.post( 
    "http://localhost:8050/get-labels",
    data = json.dumps( {
        "audio_id":audio_id,
        "annotated_areas":annotated_areas,
        "human_labels":human_labels,
        "model_name":"whisperseg-base-debug-v1.0",
        "min_frequency": 0
    }),
    headers = {"content-type":"application/json"}
).json()
```


```python
out_df = pd.DataFrame( response["labels"])
out_df["filename"]="audio.wav"
out_df["channelIndex"] = 0
out_df = out_df[["onset", "offset", "species", "individual", "clustername", "filename", "channelIndex"]]
```


```python
out_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>onset</th>
      <th>offset</th>
      <th>species</th>
      <th>individual</th>
      <th>clustername</th>
      <th>filename</th>
      <th>channelIndex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.121816</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>U</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.178039</td>
      <td>0.271743</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>U</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.290484</td>
      <td>0.393559</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>U</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.440411</td>
      <td>0.599709</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>A</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.599709</td>
      <td>0.862082</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>B</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>411</th>
      <td>204.100000</td>
      <td>204.310000</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>C</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>412</th>
      <td>204.310000</td>
      <td>204.513000</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>A</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>413</th>
      <td>204.520000</td>
      <td>204.773000</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>B</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>204.847000</td>
      <td>205.027000</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>A</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
    <tr>
      <th>415</th>
      <td>205.030000</td>
      <td>205.300000</td>
      <td>zebra finch</td>
      <td>bird1</td>
      <td>B</td>
      <td>audio.wav</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>416 rows × 7 columns</p>
</div>




```python
out_df.to_csv("example_audios/human-in-the-loop-training-example/pred_annotations.csv", index = False)
```

This prediction can be rendered on the frontend.


```python

```
