# -*- coding: utf-8 -*-
import os
import re
import sys
import random # Import random for seeding if needed elsewhere

import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.common import tokenize_by_CJK_char
from indextts.vqvae.xtts_dvae import DiscreteVAE

from indextts.utils.front import TextNormalizer # Keep original import

class IndexTTS:
    def __init__(self, cfg_path='checkpoints/config.yaml', model_dir='checkpoints', is_fp16=True):
        self.cfg = OmegaConf.load(cfg_path)
        self.device = 'cuda:0'
        self.model_dir = model_dir
        self.is_fp16 = is_fp16
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        if self.is_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = None
        self.dvae = DiscreteVAE(**self.cfg.vqvae)
        self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        load_checkpoint(self.dvae, self.dvae_path)
        self.dvae = self.dvae.to(self.device)
        if self.is_fp16:
            self.dvae.eval().half()
        else:
            self.dvae.eval()
        print(">> vqvae weights restored from:", self.dvae_path)

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            self.gpt.post_init_gpt2_config(use_deepspeed=True, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

        self.bigvgan = Generator(self.cfg.bigvgan)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location='cpu')
        self.bigvgan.load_state_dict(vocoder_dict['generator'])
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset['bpe_model']) # Use updated path from config
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")

    def preprocess_text(self, text):
        # Returns text after basic punctuation replacement,
        # as complex normalization is disabled in front.py
        return self.normalizer.infer(text)

    def remove_long_silence(self, codes, silent_token=52, max_consecutive=30):
        # (Keep the same silence removal logic as before)
        code_lens = []
        codes_list = []
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if self.cfg.gpt.stop_mel_token not in code:
                len_ = len(code)
            else:
                stop_indices = (code == self.stop_mel_token).nonzero(as_tuple=False)
                if len(stop_indices) > 0:
                    len_ = stop_indices[0].item()
                else:
                     len_ = len(code)

            code_to_check = code[:len_]
            count = torch.sum(code_to_check == silent_token).item()

            if count > max_consecutive:
                code_list = code_to_check.cpu().tolist()
                ncode = []
                n = 0
                for k in range(len_):
                    if code_list[k] != silent_token:
                        ncode.append(code_list[k])
                        n = 0
                    elif code_list[k] == silent_token and n < 10:
                        ncode.append(code_list[k])
                        n += 1
                new_len = len(ncode)
                if new_len > 0:
                    ncode_tensor = torch.LongTensor(ncode).to(self.device)
                    codes_list.append(ncode_tensor)
                    code_lens.append(new_len)
                    isfix = True
                else:
                    codes_list.append(code_to_check.to(self.device))
                    code_lens.append(len_)
            else:
                codes_list.append(code_to_check.to(self.device))
                code_lens.append(len_)

        if isfix and codes_list:
            codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
        elif codes_list:
             codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
        else:
            return torch.empty((0,0), dtype=torch.long, device=self.device), torch.empty((0,), dtype=torch.long, device=self.device)

        code_lens_tensor = torch.LongTensor(code_lens).to(self.device)
        return codes, code_lens_tensor


    # --- MODIFIED infer signature to include seed ---
    def infer(self, audio_prompt, text, output_path,
              temperature: float = 1.0,
              top_p: float = 0.8,
              top_k: int = 30,
              # --- New Parameter ---
              seed: int = -1): # Add seed parameter, default -1 for random
        print(f"origin text:{text}")
        text = self.preprocess_text(text)
        print(f"normalized text:{text}")

        # --- Set Seed ---
        if seed != -1 and isinstance(seed, int):
            print(f"Setting seed: {seed}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # For multi-GPU, though likely not needed here
            random.seed(seed) # If any other libraries use random
            # Note: numpy seeding might also be needed if numpy randomness is used internally
            # import numpy as np
            # np.random.seed(seed)
        else:
            print("Using random seed.")
            # Optional: If you want to *ensure* randomness even if called multiple times without seed
            # torch.seed() # Reseed torch from system entropy if no seed provided

        # --- Load reference audio (unchanged) ---
        audio, sr = torchaudio.load(audio_prompt)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
        print(f"cond_mel shape: {cond_mel.shape}")
        auto_conditioning = cond_mel

        # --- Tokenizer and Sentence Splitting (unchanged) ---
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(self.bpe_path)
        punctuation = ["!", "?", ".", ";", "！", "？", "。", "；"]
        pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
        sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
        print(sentences)

        # --- Hardcoded generation parameters (keep defaults, use args) ---
        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 3 # Note: Sampling (temp/top_p/top_k) usually used when num_beams=1
        repetition_penalty = 10.0
        max_mel_tokens = 600
        sampling_rate = 24000
        # lang logic might need review depending on actual use
        lang = "EN"
        lang = "ZH"
        wavs = []

        # --- Loop through sentences (calls gpt.inference_speech) ---
        for sent in sentences:
            print(sent)
            cleand_text = tokenize_by_CJK_char(sent)
            print(cleand_text)
            text_tokens = torch.IntTensor(tokenizer.encode(cleand_text)).unsqueeze(0).to(self.device)
            text_tokens = text_tokens.to(self.device)
            print(text_tokens)
            print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
            text_token_syms = [tokenizer.IdToPiece(idx) for idx in text_tokens[0].tolist()]
            print(text_token_syms)

            # --- Critical: Re-seed *inside* the loop if you want EACH sentence
            # --- segment potentially seeded differently when generating multiple
            # --- versions of the *same* sentence. If generating one version of
            # --- multiple sentences, seeding outside the loop is usually fine.
            # --- For multi-version generation, seeding just before the call is best.
            if seed != -1 and isinstance(seed, int):
                 print(f"Re-setting seed to {seed} for sentence generation.")
                 torch.manual_seed(seed)
                 torch.cuda.manual_seed_all(seed)
                 random.seed(seed)

            with torch.no_grad():
                # Generate codes using sampling parameters
                if self.is_fp16:
                    with torch.cuda.amp.autocast(enabled=self.dtype is not None, dtype=self.dtype):
                        codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                          cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                                          do_sample=True, top_p=top_p, top_k=top_k, temperature=temperature,
                                                          num_return_sequences=autoregressive_batch_size, length_penalty=length_penalty,
                                                          num_beams=num_beams, repetition_penalty=repetition_penalty, max_generate_length=max_mel_tokens)
                else:
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                      cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                                      do_sample=True, top_p=top_p, top_k=top_k, temperature=temperature,
                                                      num_return_sequences=autoregressive_batch_size, length_penalty=length_penalty,
                                                      num_beams=num_beams, repetition_penalty=repetition_penalty, max_generate_length=max_mel_tokens)

                # Process codes (unchanged)
                print(codes, type(codes))
                print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                print(codes, type(codes))
                print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                print(f"code len: {code_lens}")

                if codes.numel() == 0 or code_lens.numel() == 0 or torch.all(code_lens <= 0):
                    print(f"Warning: Skipping empty segment for sentence: '{sent}' after silence removal.")
                    continue

                # Vocoder generation (unchanged)
                if self.is_fp16:
                    with torch.cuda.amp.autocast(enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                     torch.tensor([text_tokens.shape[-1]], device=self.device), codes.to(self.device),
                                     code_lens.to(self.device)*self.gpt.mel_length_compression,
                                     cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=self.device),
                                     return_latent=True, clip_inputs=False)
                        latent = latent.transpose(1, 2)
                        wav, _ = self.bigvgan(latent.transpose(1, 2), auto_conditioning.transpose(1, 2))
                        wav = wav.squeeze(1).cpu()
                else:
                    latent = \
                        self.gpt(auto_conditioning, text_tokens,
                                 torch.tensor([text_tokens.shape[-1]], device=self.device), codes.to(self.device),
                                 code_lens.to(self.device)*self.gpt.mel_length_compression,
                                 cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=self.device),
                                 return_latent=True, clip_inputs=False)
                    latent = latent.transpose(1, 2)
                    wav, _ = self.bigvgan(latent.transpose(1, 2), auto_conditioning.transpose(1, 2))
                    wav = wav.squeeze(1).cpu()

                # Final processing (unchanged)
                wav = 32767 * wav
                torch.clip(wav, -32767.0, 32767.0)
                print(f"wav shape: {wav.shape}")
                wavs.append(wav)

        # Concatenate and save (unchanged)
        if not wavs:
            print("Error: No audio generated for any sentence.")
            silent_wav = torch.zeros((1, 100), dtype=torch.int16)
            torchaudio.save(output_path, silent_wav, sampling_rate)
            return

        wav = torch.cat(wavs, dim=1)
        torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)


if __name__ == "__main__":
    prompt_wav="test_data/input.wav"
    text="There is a vehicle arriving in dock number 7?"

    tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True)

    print("\n--- Inferring with default seed ---")
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen_default_seed.wav")

    print("\n--- Inferring with seed 1234 ---")
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen_seed_1234_a.wav", seed=1234)

    print("\n--- Inferring with seed 1234 again (should be same) ---")
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen_seed_1234_b.wav", seed=1234)

    print("\n--- Inferring with seed 5678 ---")
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen_seed_5678.wav", seed=5678)