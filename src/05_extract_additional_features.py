"""
Step 5: 多模型特征提取
Extract Additional Model Features for Multi-Model Robustness Analysis

为每个模态提取额外的特征表示，以回应审稿人R2的核心意见：
"I don't think you can make any claim about modality with just one model."

原始特征（Algonauts官方）：
  - Visual: SlowFast 3D ResNet → 8192D
  - Audio: MFCC → 20D
  - Language: BERT → 768D

新增特征：
  - Visual: CLIP ViT-B/32 → 512D
  - Audio: Wav2Vec2-base → 768D
  - Language: GPT-2 → 768D

每组特征 → PCA降维 → 保存为与官方特征相同的格式。
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# TR duration in seconds
TR_DURATION = 1.49


# ============================================================
# Visual Feature Extraction: CLIP ViT-B/32
# ============================================================

class CLIPVisualExtractor:
    """Extract visual features using CLIP ViT-B/32."""

    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.preprocess = None

    def load_model(self):
        from transformers import CLIPModel, CLIPProcessor
        print("  Loading CLIP ViT-B/32...")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.model.eval()
        print("  CLIP loaded. Feature dim: 512")

    def extract_from_video(self, video_path, n_trs, batch_size=64):
        """Extract one feature vector per TR from video using ffmpeg for fast frame extraction."""
        import subprocess
        import tempfile
        from PIL import Image
        import io

        if self.model is None:
            self.load_model()

        # Use ffmpeg to extract frames at TR rate (much faster than OpenCV seeking)
        fps_out = 1.0 / TR_DURATION  # ~0.671 fps

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract frames at TR intervals using ffmpeg
            subprocess.run([
                'ffmpeg', '-i', str(video_path),
                '-vf', f'fps={fps_out}',
                '-q:v', '2',  # high quality JPEG
                '-y', '-loglevel', 'quiet',
                os.path.join(tmpdir, 'frame_%05d.jpg')
            ], check=True, timeout=300)

            # Load extracted frames
            frame_files = sorted([
                f for f in os.listdir(tmpdir) if f.endswith('.jpg')])

            tr_frames = []
            for i in range(min(n_trs, len(frame_files))):
                img = Image.open(os.path.join(tmpdir, frame_files[i])).convert('RGB')
                tr_frames.append(np.array(img))

        # Pad if needed
        while len(tr_frames) < n_trs:
            tr_frames.append(tr_frames[-1] if tr_frames else
                             np.zeros((224, 224, 3), dtype=np.uint8))

        # Extract features in batches
        features = []
        with torch.no_grad():
            for i in range(0, len(tr_frames), batch_size):
                batch = tr_frames[i:i+batch_size]
                inputs = self.processor(images=batch, return_tensors="pt", padding=True)
                pixel_values = inputs['pixel_values'].to(self.device)
                outputs = self.model.vision_model(pixel_values=pixel_values)
                batch_feat = outputs.pooler_output
                features.append(batch_feat.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features[:n_trs]


# ============================================================
# Audio Feature Extraction: Wav2Vec2
# ============================================================

class Wav2Vec2AudioExtractor:
    """Extract audio features using Wav2Vec2-base."""

    def __init__(self, device='cuda', sr=16000):
        self.device = device
        self.sr = sr
        self.model = None

    def load_model(self):
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        print("  Loading Wav2Vec2-base...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.model.eval()
        print("  Wav2Vec2 loaded. Feature dim: 768")

    def extract_from_video(self, video_path, n_trs):
        """Extract audio features from video, one vector per TR."""
        import subprocess
        import tempfile
        import soundfile as sf

        if self.model is None:
            self.load_model()

        # Extract audio from video to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            subprocess.run([
                'ffmpeg', '-i', str(video_path), '-ac', '1', '-ar', str(self.sr),
                '-vn', '-y', '-loglevel', 'quiet', tmp_path
            ], check=True)

            audio, sr = sf.read(tmp_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
        finally:
            os.unlink(tmp_path)

        # Process audio in TR-aligned windows
        samples_per_tr = int(TR_DURATION * self.sr)
        features = []

        with torch.no_grad():
            for tr_idx in range(n_trs):
                start = tr_idx * samples_per_tr
                end = start + samples_per_tr

                if end > len(audio):
                    # Pad with zeros
                    chunk = np.zeros(samples_per_tr)
                    valid = min(len(audio) - start, samples_per_tr)
                    if valid > 0:
                        chunk[:valid] = audio[start:start+valid]
                else:
                    chunk = audio[start:end]

                inputs = self.processor(
                    chunk, sampling_rate=self.sr, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                # Mean pool over time
                hidden = outputs.last_hidden_state.mean(dim=1)
                features.append(hidden.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features[:n_trs]


# ============================================================
# Language Feature Extraction: GPT-2
# ============================================================

class GPT2LanguageExtractor:
    """Extract language features using GPT-2."""

    def __init__(self, device='cuda'):
        self.device = device
        self.model = None

    def load_model(self):
        from transformers import GPT2Model, GPT2Tokenizer
        print("  Loading GPT-2...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained("gpt2").to(self.device)
        self.model.eval()
        print("  GPT-2 loaded. Feature dim: 768")

    def extract_from_transcript(self, transcript_path, n_trs):
        """Extract language features from transcript, one vector per TR."""
        import csv

        if self.model is None:
            self.load_model()

        # Read transcript (TSV, one row per TR)
        texts = []
        with open(transcript_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # Skip header
            for row in reader:
                text = row[0].strip() if row and row[0].strip() else ""
                texts.append(text)

        # Pad or truncate to n_trs
        while len(texts) < n_trs:
            texts.append("")
        texts = texts[:n_trs]

        # Extract features
        features = []
        with torch.no_grad():
            for text in texts:
                if not text:
                    features.append(np.zeros(768))
                    continue

                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=128, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                # Mean pool over tokens
                hidden = outputs.last_hidden_state.mean(dim=1)
                features.append(hidden.cpu().numpy().flatten())

        features = np.array(features)
        return features[:n_trs]


# ============================================================
# Main Pipeline
# ============================================================

class AdditionalFeatureExtractor:
    """完整的多模型特征提取流程"""

    def __init__(self, project_dir, output_dir=None, pca_dim=250, pca_dim_audio=20):
        self.project_dir = project_dir
        self.pca_dim = pca_dim
        self.pca_dim_audio = pca_dim_audio
        self.output_dir = output_dir or os.path.join(
            project_dir, 'data', 'features', 'additional_features')
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Reference to official PCA features for segment names and TR counts
        self.official_pca_dir = os.path.join(
            project_dir, 'data', 'features', 'official_stimulus_features',
            'pca', 'friends_movie10')

        # Excluded samples (same as encoding pipeline)
        self.excluded_samples_start = 5
        self.excluded_samples_end = 5

    def get_segment_info(self):
        """获取训练/测试段的名称和TR数"""
        # Load official features to get segment names and lengths
        train_data = np.load(
            os.path.join(self.official_pca_dir, 'visual', 'features_train.npy'),
            allow_pickle=True).item()
        test_data = np.load(
            os.path.join(self.official_pca_dir, 'visual', 'features_test.npy'),
            allow_pickle=True).item()

        train_segments = {k: v.shape[0] for k, v in train_data.items()}
        test_segments = {k: v.shape[0] for k, v in test_data.items()}

        return train_segments, test_segments

    def get_video_path(self, segment_name):
        """根据段名获取视频路径"""
        stimuli_dir = os.path.join(self.project_dir, 'data', 'stimuli', 'movies')

        if segment_name.startswith('s'):
            # Friends: s01e01a → friends/s1/friends_s01e01a.mkv
            season = segment_name[:3]  # s01
            season_num = int(season[1:])
            video_name = f'friends_{segment_name}.mkv'
            return os.path.join(stimuli_dir, 'friends', f's{season_num}', video_name)
        else:
            # Movie10: bourne01a → movie10/bourne/bourne01a.mkv or similar
            # Parse movie name
            for movie in ['bourne', 'figures', 'life', 'wolf']:
                if segment_name.startswith(movie):
                    video_name = f'{segment_name}.mkv'
                    return os.path.join(stimuli_dir, 'movie10', movie, video_name)

        return None

    def get_transcript_path(self, segment_name):
        """根据段名获取转录文件路径"""
        transcripts_dir = os.path.join(self.project_dir, 'data', 'stimuli', 'transcripts')

        if segment_name.startswith('s'):
            season = segment_name[:3]
            season_num = int(season[1:])
            transcript_name = f'friends_{segment_name}.tsv'
            return os.path.join(transcripts_dir, 'friends', f's{season_num}', transcript_name)
        else:
            for movie in ['bourne', 'figures', 'life', 'wolf']:
                if segment_name.startswith(movie):
                    # Movie10 transcripts: movie10/bourne/movie10_bourne01.tsv
                    transcript_name = f'movie10_{segment_name}.tsv'
                    return os.path.join(
                        transcripts_dir, 'movie10', movie, transcript_name)

        return None

    def extract_modality(self, modality, model_name, extractor, segments, split_name):
        """提取单个模态的特征"""
        print(f"\n{'='*60}")
        print(f"Extracting {modality} features using {model_name}")
        print(f"Split: {split_name}, Segments: {len(segments)}")
        print(f"{'='*60}")

        features_dict = {}
        failed = []

        for seg_name, n_trs in tqdm(segments.items(), desc=f"{model_name}/{split_name}"):
            try:
                if modality == 'visual':
                    video_path = self.get_video_path(seg_name)
                    if video_path and os.path.exists(video_path):
                        feats = extractor.extract_from_video(video_path, n_trs)
                    else:
                        print(f"  Warning: video not found for {seg_name}: {video_path}")
                        failed.append(seg_name)
                        continue

                elif modality == 'audio':
                    video_path = self.get_video_path(seg_name)
                    if video_path and os.path.exists(video_path):
                        feats = extractor.extract_from_video(video_path, n_trs)
                    else:
                        failed.append(seg_name)
                        continue

                elif modality == 'language':
                    transcript_path = self.get_transcript_path(seg_name)
                    if transcript_path and os.path.exists(transcript_path):
                        feats = extractor.extract_from_transcript(transcript_path, n_trs)
                    else:
                        failed.append(seg_name)
                        continue

                features_dict[seg_name] = feats

            except Exception as e:
                print(f"  Error extracting {seg_name}: {e}")
                failed.append(seg_name)

        if failed:
            print(f"  Failed segments: {len(failed)}/{len(segments)}")

        return features_dict

    def apply_pca_and_save(self, train_features, test_features, modality, model_name):
        """Apply PCA and save in official format."""
        pca_dim = self.pca_dim_audio if modality == 'audio' else self.pca_dim

        # Concatenate all training features for PCA fit
        all_train = np.concatenate([v for v in train_features.values()], axis=0)

        # Fit scaler and PCA
        scaler = StandardScaler()
        all_train_scaled = scaler.fit_transform(all_train)

        actual_dim = min(pca_dim, all_train_scaled.shape[1], all_train_scaled.shape[0])
        pca = PCA(n_components=actual_dim)
        pca.fit(all_train_scaled)

        print(f"  PCA: {all_train_scaled.shape[1]}D → {actual_dim}D "
              f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

        # Transform features
        def transform_dict(feat_dict):
            result = {}
            for seg_name, feats in feat_dict.items():
                scaled = scaler.transform(feats)
                result[seg_name] = pca.transform(scaled).astype(np.float32)
            return result

        train_pca = transform_dict(train_features)
        test_pca = transform_dict(test_features)

        # Save
        save_dir = os.path.join(self.output_dir, model_name, 'pca', modality)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'features_train.npy'), train_pca)
        np.save(os.path.join(save_dir, 'features_test.npy'), test_pca)
        np.save(os.path.join(save_dir, 'pca_param.npy'), {
            'components_': pca.components_,
            'mean_': pca.mean_,
            'explained_variance_': pca.explained_variance_,
            'explained_variance_ratio_': pca.explained_variance_ratio_,
            'n_components': actual_dim,
        })
        np.save(os.path.join(save_dir, 'scaler_param.npy'), {
            'mean_': scaler.mean_,
            'scale_': scaler.scale_,
        })

        print(f"  Saved to {save_dir}")
        return train_pca, test_pca

    def run(self, modalities=None, skip_existing=True):
        """运行完整的特征提取流程"""
        if modalities is None:
            modalities = ['visual', 'audio', 'language']

        train_segments, test_segments = self.get_segment_info()
        print(f"Train segments: {len(train_segments)}, Test segments: {len(test_segments)}")

        # Define model-modality pairs
        extractors = {
            'visual': ('clip_vitb32', CLIPVisualExtractor(self.device)),
            'audio': ('wav2vec2_base', Wav2Vec2AudioExtractor(self.device)),
            'language': ('gpt2', GPT2LanguageExtractor(self.device)),
        }

        results = {}

        for modality in modalities:
            model_name, extractor = extractors[modality]

            # Check if already extracted
            save_dir = os.path.join(self.output_dir, model_name, 'pca', modality)
            if (skip_existing and os.path.exists(save_dir) and
                    os.path.exists(os.path.join(save_dir, 'features_train.npy'))):
                print(f"\n  Skipping {modality}/{model_name} - already exists")
                continue

            # Extract raw features
            train_feats = self.extract_modality(
                modality, model_name, extractor, train_segments, 'train')
            test_feats = self.extract_modality(
                modality, model_name, extractor, test_segments, 'test')

            if not train_feats:
                print(f"  ERROR: No features extracted for {modality}/{model_name}")
                continue

            # PCA and save
            self.apply_pca_and_save(train_feats, test_feats, modality, model_name)
            results[modality] = model_name

        # Save extraction summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models': results,
            'pca_dim': self.pca_dim,
            'pca_dim_audio': self.pca_dim_audio,
            'n_train_segments': len(train_segments),
            'n_test_segments': len(test_segments),
        }
        with open(os.path.join(self.output_dir, 'extraction_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*60)
        print("Feature extraction complete!")
        print(f"Output: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Extract additional model features for multi-model robustness')
    parser.add_argument('--project_dir', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help='Project root directory')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--modalities', default='visual,audio,language',
                        help='Modalities to extract (comma-separated)')
    parser.add_argument('--pca_dim', type=int, default=250,
                        help='PCA target dimension for visual/language')
    parser.add_argument('--pca_dim_audio', type=int, default=20,
                        help='PCA target dimension for audio')
    parser.add_argument('--no_skip', action='store_true',
                        help='Re-extract even if features already exist')

    args = parser.parse_args()
    modalities = [m.strip() for m in args.modalities.split(',')]

    extractor = AdditionalFeatureExtractor(
        project_dir=args.project_dir,
        output_dir=args.output_dir,
        pca_dim=args.pca_dim,
        pca_dim_audio=args.pca_dim_audio,
    )

    extractor.run(modalities=modalities, skip_existing=not args.no_skip)


if __name__ == "__main__":
    main()
