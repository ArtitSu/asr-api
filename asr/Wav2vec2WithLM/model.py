import json

from transformers import AutoProcessor
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
import torch
import torchaudio

class Model:
    def __init__(self):
        self.model = Wav2Vec2ForCTC.from_pretrained("wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # your LM path
        your_lm = "asr/Wav2vec2WithLM/lm.arpa"

        # get the tokenizer
        processor = AutoProcessor.from_pretrained("wannaphong/wav2vec2-large-xlsr-53-th-cv8-newmm")
        vocab_dict = processor.tokenizer.get_vocab()
        sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

        # build the decoder with your LM, you can configure your LM weight by passing alpha and beta.
        # alpha: weight for language model during shallow fusion
        # beta: weight for length score adjustment of during scoring
        decoder = build_ctcdecoder(
            labels=list(sorted_vocab_dict.keys()),
            kenlm_model_path=your_lm,
            alpha=0.5,
            beta=1.5
        )
        
        self.processor_with_lm = Wav2Vec2ProcessorWithLM(
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            decoder=decoder
        )

    def predict(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        
        resampler = torchaudio.transforms.Resample(sr,16000)
        audio = resampler(audio)[0]
        inputs = self.processor_with_lm(audio, sampling_rate=16000, return_tensors="pt")

        # get logits from the model
        with torch.no_grad():
          logits = self.model(**inputs).logits

        # decode to transcription
        transcription = self.processor_with_lm.batch_decode(logits.numpy()).text
        transcription = transcription[0].lower()

        return transcription

model = Model()

def get_model():
    return model
