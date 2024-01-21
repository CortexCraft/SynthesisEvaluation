from pathlib import Path
import numpy as np
import torch
import librosa
import soundfile
from speechbrain.pretrained import EncoderASR
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


_model = None # type: Wav2Vec2ForCTC
_processor = None # type: Wav2Vec2Processor
_device = None # type: torch.device


def load_model(device=None):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    global _model, _processor, _device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _device = torch.device(device)
    elif isinstance(device, str):
        _device = torch.device(device)
        
    # load model and processor
    _processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    _model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

    _model.to(_device)
    _model.eval()
    print("Loaded ASR encoder.")
    
    
def is_loaded():
    return _model is not None

@torch.no_grad()
def generate_token_predictions(wav_file, sample_rate:int=16000):
    """
    Args:
        wav_file: Path to audio wav file
        sample_rate: target sample rate
    
    returns:
        1D token IDs vector (T,)
    """
    wave, sr = librosa.load(wav_file, sr=sample_rate)
    # tokenize
    input_values = _processor(wave, return_tensors="pt", sampling_rate = sr, padding="longest").input_values.to(_device)

    # retrieve logits
    logits = _model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = processor.batch_decode(predicted_ids)

    return predicted_ids.squeeze(0).cpu().numpy()