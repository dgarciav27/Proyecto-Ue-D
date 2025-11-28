import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import ViTImageProcessor
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ViTAudioDataset(Dataset):
    
    def __init__(self, args, mode, load_info=False):
        self.args = args
        self.mode = mode
        self.load_info = load_info
        self.labels = args.labels

        self.audio_path = None
        self.annot_path = None
        self.annotations = None
        self._init_path_and_annot()

        # Initialize Vit preprocesor
        self.processor = ViTImageProcessor.from_pretrained(args.model_path)  # Model path
        
        # Transformations for spectograms
        self.spectroT = torchaudio.transforms.Spectrogram(
            n_fft=self.args.n_fft,
            win_length=self.args.win_size,
            hop_length=int(self.args.win_size - (self.args.win_size * self.args.overlap / 100)))

        self.amp_to_dbT = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=1000)

    def _init_path_and_annot(self):
        self.audio_path = os.path.join(self.args.data_root, self.mode, 'audio')

        if self.mode == 'train':
            self.annot_path = os.path.join(self.args.data_root, self.mode, self.args.train_annot)
            if os.path.isfile(self.annot_path):
                self.annotations = pd.read_csv(self.annot_path)
            else:
                final_df = pd.DataFrame()
                for file in [f for f in os.listdir(self.annot_path) if f.endswith('.csv')]:
                    df_tmp = pd.read_csv(os.path.join(self.annot_path, file))
                    final_df = pd.concat([final_df, df_tmp], ignore_index=True)
                self.annotations = final_df

        elif self.mode == 'validation':
            self.annot_path = os.path.join(self.args.data_root, self.mode, self.args.val_annot)
            if os.path.isfile(self.annot_path):
                self.annotations = pd.read_csv(self.annot_path)
            else:
                final_df = pd.DataFrame()
                for file in [f for f in os.listdir(self.annot_path) if f.endswith('.csv')]:
                    df_tmp = pd.read_csv(os.path.join(self.annot_path, file))
                    final_df = pd.concat([final_df, df_tmp], ignore_index=True)
                self.annotations = final_df

        elif self.mode == 'test':
            self.annot_path = os.path.join(self.args.data_root, self.mode, self.args.test_annot)
            if os.path.isfile(self.annot_path):
                self.annotations = pd.read_csv(self.annot_path)
            else:
                final_df = pd.DataFrame()
                for file in [f for f in os.listdir(self.annot_path) if f.endswith('.csv')]:
                    df_tmp = pd.read_csv(os.path.join(self.annot_path, file))
                    final_df = pd.concat([final_df, df_tmp], ignore_index=True)
                self.annotations = final_df

    def gen_spectro_normalized(self, audio_path, start_offset, resample_sr, duration, db_threshold=30):
        info = torchaudio.info(audio_path)

        signal, orig_sr = torchaudio.load(audio_path,
                                          format='wav',
                                          frame_offset=int(start_offset * info.sample_rate),
                                          num_frames=int(duration * info.sample_rate))

        if orig_sr != resample_sr:
            signal = torchaudio.functional.resample(signal, orig_freq=orig_sr, new_freq=resample_sr)
        signal = signal / signal.std()

        spectro = self.amp_to_dbT(self.spectroT(signal))
        spectro.clamp_(-db_threshold, db_threshold)
        normalized_spectro = (spectro + db_threshold) / (2 * db_threshold)
        
        return normalized_spectro

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        dataset = self.annotations.iloc[idx, self.annotations.columns.get_loc('dataset')]
        filename = self.annotations.iloc[idx, self.annotations.columns.get_loc('filename')]
        audio_path = os.path.join(self.audio_path, dataset, filename)
        start_offset = self.annotations.iloc[idx, self.annotations.columns.get_loc('start_offset')]

        # Generate spectograms
        spectro = self.gen_spectro_normalized(audio_path,
                                              start_offset,
                                              resample_sr=self.args.sample_rate,
                                              duration=self.args.duration)
        
        # Convert to Vit format
        # 1. Expand to 3 channels (RGB)
        spectro_3ch = spectro.expand(3, -1, -1)
        
        # 2. COnverto to PIL Image for Vit processor
        # First normalize to [0, 255] then convert to uint8
        spectro_np = (spectro_3ch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(spectro_np)
        
        # 3. Processing with Vit processor
        inputs = self.processor(pil_image, return_tensors="pt")
        
        # Obtain labels
        idx_lab = [self.annotations.columns.get_loc(lab) for lab in self.labels]
        labels = self.annotations.iloc[idx, idx_lab].to_numpy().astype(float)
        labels = torch.Tensor(labels)

        if self.load_info:
            return (dataset, filename, start_offset), inputs, labels
        else:
            return inputs, labels