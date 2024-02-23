# Installation of environment

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

# Start service
In the main folder, run
```bash
python backend.py
```