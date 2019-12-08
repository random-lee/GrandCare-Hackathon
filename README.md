# GrandCare (Hackathon Dec 2019)

This code is based on the [PyTorch audio asr-demo](https:////github.com/pytorch/audio/tree/master/examples/interactive_asr)


to run you needs the following libraries:
* pyaudio
* torchaudio
* pytorch
* librosa
* flask
* fairseq (clone the pytorch github repository)

to run you needs the following models
```
# Install dictionary, sentence piece model, and model
wget -O ./data/dict.txt https://download.pytorch.org/models/audio/dict.txt
wget -O ./data/spm.model https://download.pytorch.org/models/audio/spm.model
wget -O ./data/model.pt https://download.pytorch.org/models/audio/checkpoint_avg_60_80.pt
```


## Run

start ngrok to connect you localhost to the internet

python3 server.py
