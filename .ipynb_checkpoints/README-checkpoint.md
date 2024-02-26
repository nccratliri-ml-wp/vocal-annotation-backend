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