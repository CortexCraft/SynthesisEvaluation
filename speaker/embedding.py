from pathlib import Path
import numpy as np
import torch
import librosa
import soundfile
from speechbrain.pretrained import EncoderClassifier

_xvec_model = None # type: EncoderClassifier
_tdnn_model = None # type: EncoderClassifier
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
    global _xvec_model, _tdnn_model, _device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _device = torch.device(device)
    elif isinstance(device, str):
        _device = torch.device(device)
        
    _tdnn_model = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                                 savedir='models/speaker_embeddings/spkrec-ecapa-voxceleb',
                                                 run_opts={'device': device})
    _xvec_model = EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb',
                                                 savedir='models/speaker_embeddings/spkrec-xvect-voxceleb',
                                                 run_opts={'device': device})
    _tdnn_model.eval()
    _xvec_model.eval()
    print("Loaded Speaker encoder.")


def is_loaded():
    return _tdnn_model is not None and _xvec_model is not None

def generate_speaker_embedding(wav_file, sample_rate:int=16000):
    """
    Args:
        wav_file: Path to audio wav file
        sample_rate: target sample rate
    
    returns:
        2D BNF vector (H, T)
    """
    wave, _ = librosa.load(wav_file, sr=16000)
    wave = torch.tensor(np.trim_zeros(wave))

    spk_emb_x = _xvec_model.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
    spk_emb_t = _tdnn_model.encode_batch(wavs=wave.unsqueeze(0)).squeeze()
    embed = torch.cat([spk_emb_x, spk_emb_t], dim=0)
    
    embed = embed.cpu().numpy()
    return embed
