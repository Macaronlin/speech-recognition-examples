import subprocess

import torch
import torchaudio
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from telegram.ext import Updater, MessageHandler, Filters
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cuda")
resampler = torchaudio.transforms.Resample(48_000, 16_000)


def voice_handler(update, context):
    file = context.bot.getFile(update.message.voice.file_id)
    src = str(update.message.voice.file_id) + '.ogg'
    dst = str(update.message.voice.file_id) + '.wav'
    file.download(src)
    process = subprocess.run(['ffmpeg', '-y', '-i', src, dst])
    if process.returncode != 0:
        raise Exception("Something went wrong")
    speech = resampler(torchaudio.load(dst)[0]).squeeze().numpy()
    input_values = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits
    os.remove(src)
    os.remove(dst)
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=processor.batch_decode(torch.argmax(logits, dim=-1))[0].lower())


updater = Updater(token='TOKEN')
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.voice, voice_handler))
updater.start_polling()
