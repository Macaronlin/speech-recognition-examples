import os
import sys

import Levenshtein as leven
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from colorama import Fore
from skimage.color import rgb2gray
from skimage.transform import rotate
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import IAMModel
from preprocessing import text_transform, valid_audio_transforms, train_audio_transforms

dev = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================= PREPARING DATASET ======================================================

if not os.path.isdir("./data"):
    os.makedirs("./data")
train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)

classes = ''.join(text_transform.index_map.values())
print(classes)
print(len(classes))
text_file = open("chars.txt", "w", encoding='utf-8')
text_file.write('\n'.join([x if x != '<SPACE>' else ' ' for x in text_transform.char_map.keys()]))
text_file.close()


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


# ================================================= MODEL ==============================================================
model = IAMModel(time_step=4096,
                 feature_size=512,
                 hidden_size=512,
                 output_size=len(classes) + 1,
                 num_rnn_layers=2)
model.to(dev)


# ================================================ TRAINING MODEL ======================================================
def fit(model, epochs, train_data_loader, valid_data_loader, lr=5e-4, wd=1e-2, betas=(0.9, 0.999)):
    best_leven = 1000
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                     weight_decay=wd, betas=betas)
    opt.zero_grad(set_to_none=False)
    len_train = len(train_data_loader)
    loss_func = nn.CTCLoss(reduction='sum', zero_infinity=True, blank=len(classes))
    for i in range(1, epochs + 1):
        # ============================================ TRAINING ========================================================
        batch_n = 1
        loss = 0
        train_levenshtein = 0
        len_levenshtein = 0
        for spectrograms, labels, input_lengths, label_lengths in tqdm(train_data_loader,
                                 position=0, leave=True,
                                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            model.train()
            spectrograms, labels = spectrograms.to(dev), labels.to(dev)
            # And the lengths are specified for each sequence to achieve masking
            # under the assumption that sequences are padded to equal lengths.
            loss = loss_func(model(spectrograms).log_softmax(2).requires_grad_(), labels, input_lengths, label_lengths)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=False)
            # ================================== TRAINING LEVENSHTEIN DISTANCE =========================================
            if batch_n > (len_train - 5):
                model.eval()
                with torch.no_grad():
                    decoded = model.beam_search_with_lm(spectrograms)
                    for j in range(0, len(decoded)):
                        # We need to find actual string somewhere in the middle of the 'targets'
                        # tensor having tensor 'lens' with known lengths
                        actual = text_transform.int_to_text(labels.cpu().numpy()[i][:label_lengths[i]].tolist())
                        train_levenshtein += leven.distance(''.join([letter for letter in decoded[j]]), actual)
                    len_levenshtein += sum(label_lengths)

            batch_n += 1

        # ============================================ VALIDATION ======================================================
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            val_levenshtein = 0
            target_lengths = 0
            for spectrograms, labels, input_lengths, label_lengths in tqdm(valid_data_loader,
                                     position=0, leave=True,
                                     file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
                spectrograms, labels = spectrograms.to(dev), labels.to(dev)
                valid_loss += loss_func(model(spectrograms), labels, input_lengths, label_lengths)
                decoded = model.beam_search_with_lm(spectrograms)
                for j in range(0, len(decoded)):
                    actual = text_transform.int_to_text(labels.cpu().numpy()[i][:label_lengths[i]].tolist())
                    val_levenshtein += leven.distance(''.join([letter for letter in decoded[j]]), actual)
                target_lengths += sum(label_lengths)

        print('epoch {}: Train Levenshtein {} | Validation Levenshtein {}'
              .format(i, train_levenshtein / len_levenshtein, val_levenshtein / target_lengths), end='\n')
        # ============================================ SAVE MODEL ======================================================
        if (val_levenshtein / target_lengths) < best_leven:
            torch.save(model.state_dict(), f=str((val_levenshtein / target_lengths) * 100).replace('.', '_') + '_' + 'model.pth')
            best_leven = val_levenshtein / target_lengths


train_batch_size = 40
validation_batch_size = 20
torch.manual_seed(7)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=lambda x: data_processing(x, 'train'), pin_memory=True)
validation_loader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=lambda x: data_processing(x, 'valid'), pin_memory=True)
print("Training...")
# model.load_state_dict(torch.load('./171_1224381060633_model.pth'))
fit(model=model, epochs=2, train_data_loader=train_loader, valid_data_loader=validation_loader)


# ============================================ TESTING =================================================================
def batch_predict(model, valid_dl, up_to):
    spectrograms, labels, input_lengths, label_lengths = iter(valid_dl).next()
    model.eval()
    spectrograms, labels = spectrograms.to(dev), labels.to(dev)
    with torch.no_grad():
        outs = model.beam_search_with_lm(spectrograms)
        for i in range(len(outs)):
            # start = sum(lens[:i])
            # end = lens[i].item()
            actual = text_transform.int_to_text(labels.cpu().numpy()[i][:label_lengths[i]].tolist())
            predicted = ''.join([letter for letter in outs[i]])
            # ============================================ SHOW IMAGE ==================================================
            # img = xb[i, :, :, :].permute(1, 2, 0).cpu().numpy()
            # img = rgb2gray(img)
            # img = rotate(img, angle=90, clip=False, resize=True)
            # f, ax = plt.subplots(1, 1)
            # mpl.rcParams["font.size"] = 8
            # ax.imshow(img, cmap='gray')
            # mpl.rcParams["font.size"] = 14
            # plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(actual))
            # plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(predicted))
            # f.set_size_inches(10, 3)
            print('actual: {}'.format(actual))
            print('predicted:   {}'.format(predicted))
            if i + 1 == up_to:
                break
    plt.show()


batch_predict(model=model, valid_dl=validation_loader, up_to=20)
