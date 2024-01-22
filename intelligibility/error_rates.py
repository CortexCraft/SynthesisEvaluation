import re
from jiwer import wer, cer
import asr as ASR


def remove_special_characters(text):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    text = re.sub(chars_to_ignore_regex, '', text).lower()

    return text

class ErrorRates(object):
    """
    Calculate WER / CER
    """
    def __init__(self):
        super(ErrorRates).__init__()
        self.asr = ASR
        self.asr.load_model()

    def get_error_rates(self, audio, reference_text):
        transcibed_text = self.asr.generate_transcription(audio)

        wer_ = wer(reference_text, transcibed_text)
        cer_ = cer(reference_text, transcibed_text)

        return wer_, cer_
