import torch
import torchaudio
from jiwer import wer
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cuda")
test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)

decoded = []
original = []

for i in range(40):
    waveform, _, utterance, _, _, _ = test_dataset[i]
    input_values = tokenizer(waveform.numpy().squeeze(0), return_tensors="pt", padding="longest").input_values
    decoded.append(tokenizer.batch_decode(torch.argmax(model(input_values.to("cuda")).logits, dim=-1))[0])
    original.append(utterance)

# 0.010767160161507403
print(wer(original, decoded))
