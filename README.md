
<div align="center">
<img src='assets/index_icon.png' width="250"/>
</div>


<h2><center>IndexTTS Workflow Studio: An Enhanced Zero-Shot TTS System with Advanced Workflow Tools</h2>

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>

## 👉🏻 IndexTTS Workflow Studio 👈🏻

[[HuggingFace Demo]](https://huggingface.co/spaces/IndexTeam/IndexTTS)   [[ModelScope Demo]](https://modelscope.cn/studios/IndexTeam/IndexTTS-Demo) \
[[Original IndexTTS Paper]](https://arxiv.org/abs/2502.05512)  [[Original IndexTTS Demos]](https://index-tts.github.io)

**IndexTTS Workflow Studio** builds upon the original **IndexTTS** (a GPT-style text-to-speech model based on XTTS and Tortoise) and adds a comprehensive user interface and workflow tools for generating, reviewing, and post-processing speech. It retains the core capabilities of the original model while providing significant enhancements for practical use cases.

This version includes major additions such as a multi-tab Gradio UI, advanced seed management, interactive audio review with speaker similarity feedback, audio concatenation, and extensive post-processing effects. See the `NOTICE` file for a detailed breakdown of modifications. *Coming Soon: Keep an eye out for DialogueLab, a related project designed to work seamlessly with the API provided by IndexTTS Workflow Studio for creating interactive dialogues (Release expected soon!)*

### Author & Contact

*   **Author:** James A Whittaker-Bent
*   **Email:** Whittakerbent@googlemail.com
*   **Development Note:** This Workflow Studio, featuring a comprehensive Gradio UI, audio post-processing pipeline, and FastAPI integration, represents a significant expansion developed in an intensive 3-day sprint, effectively leveraging modern AI development tools.
*   Feel free to reach out regarding potential collaborations or questions about the Workflow Studio additions.

*Original IndexTTS Contact (for questions about the core model):*
*   QQ Group: 553460296
*   Discord: https://discord.gg/uT32E7KDmy

## License and Attribution

This project, **IndexTTS Workflow Studio**, is licensed under the **Apache License, Version 2.0** (see the `LICENSE` file).

It is based on the original **IndexTTS** project by Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang (Repository: https://github.com/index-tts/index-tts), which is also licensed under Apache 2.0.

Significant modifications and additions were made by James A Whittaker-Bent. Please see the `NOTICE` file for detailed attribution, a summary of changes, and information regarding the licensing of the original pre-trained models (see `INDEX_MODEL_LICENSE`).
## 📣 Updates

- `2025/03/25` 🔥🔥 We release the model parameters and inference code.
- `2025/02/12` 🔥 We submitted our paper on arXiv, and released our demos and test sets.

## 🖥️ Method

The overview of IndexTTS is shown as follows.

<picture>
  <img src="assets/IndexTTS.png"  width="800"/>
</picture>


The main improvements and contributions are summarized as follows:
 - In Chinese scenarios, we have introduced a character-pinyin hybrid modeling approach. This allows for quick correction of mispronounced characters.
 - **IndexTTS** incorporate a conformer conditioning encoder and a BigVGAN2-based speechcode decoder. This improves training stability, voice timbre similarity, and sound quality.
 - We release all test sets here, including those for polysyllabic words, subjective and objective test sets.



## Model Download
| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [Original IndexTTS Models (HuggingFace)](https://huggingface.co/IndexTeam/Index-TTS) | [Original IndexTTS Models (ModelScope)](https://modelscope.cn/models/IndexTeam/Index-TTS) |


## 📑 Evaluation

**Word Error Rate (WER) Results for IndexTTS and Baseline Models**


|    **Model**    | **aishell1_test** | **commonvoice_20_test_zh** | **commonvoice_20_test_en** | **librispeech_test_clean** |  **avg** |
|:---------------:|:-----------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------:|
|    **Human**    |        2.0        |            9.5             |            10.0            |            2.4             |   5.1    |
| **CosyVoice 2** |        1.8        |            9.1             |            7.3             |            4.9             |   5.9    |
|    **F5TTS**    |        3.9        |            11.7            |            5.4             |            7.8             |   8.2    |
|  **Fishspeech** |        2.4        |            11.4            |            8.8             |            8.0             |   8.3    |
|  **FireRedTTS** |        2.2        |            11.0            |            16.3            |            5.7             |   7.7    |
|     **XTTS**    |        3.0        |            11.4            |            7.1             |            3.5             |   6.0    |
|   **IndexTTS**  |      **1.3**     |          **7.0**          |          **5.3**          |          **2.1**          | **3.7** |


**Speaker Similarity (SS) Results for IndexTTS and Baseline Models**

|    **Model**    | **aishell1_test** | **commonvoice_20_test_zh** | **commonvoice_20_test_en** | **librispeech_test_clean** |  **avg**  |
|:---------------:|:-----------------:|:--------------------------:|:--------------------------:|:--------------------------:|:---------:|
|    **Human**    |       0.846       |            0.809           |            0.820           |            0.858           |   0.836   |
| **CosyVoice 2** |     **0.796**     |            0.743           |            0.742           |          **0.837**         | **0.788** |
|    **F5TTS**    |       0.743       |          **0.747**         |            0.746           |            0.828           |   0.779   |
|  **Fishspeech** |       0.488       |            0.552           |            0.622           |            0.701           |   0.612   |
|  **FireRedTTS** |       0.579       |            0.593           |            0.587           |            0.698           |   0.631   |
|     **XTTS**    |       0.573       |            0.586           |            0.648           |            0.761           |   0.663   |
|   **IndexTTS**  |       0.744       |            0.742           |          **0.758**         |            0.823           |   0.776   |



**MOS Scores for Zero-Shot Cloned Voice**

| **Model**       | **Prosody** | **Timbre** | **Quality** |  **AVG**  |
|-----------------|:-----------:|:----------:|:-----------:|:---------:|
| **CosyVoice 2** |    3.67     |    4.05    |    3.73     |   3.81    |
| **F5TTS**       |    3.56     |    3.88    |    3.56     |   3.66    |
| **Fishspeech**  |    3.40     |    3.63    |    3.69     |   3.57    |
| **FireRedTTS**  |    3.79     |    3.72    |    3.60     |   3.70    |
| **XTTS**        |    3.23     |    2.99    |    3.10     |   3.11    |
| **IndexTTS**    |    **3.79**     |    **4.20**    |    **4.05**     |   **4.01**    |


## Usage Instructions

**Important Note on Models:** This repository contains the code for IndexTTS Workflow Studio. The large pre-trained model files (`.pth`, `.vocab`, `.model`) are **not** included here. You must download them separately from the original IndexTTS project resources (HuggingFace or ModelScope, see Model Download section above) and place them in the `checkpoints` directory. These models are subject to the `INDEX_MODEL_LICENSE` file included in this repository.

### Environment Setup

**Prerequisites:**
*   Git
*   Conda (Miniconda or Anaconda)
*   Python 3.10

1.  **Clone this repository:**
    ```bash
            # Replace with your actual repository URL after uploading to GitHub
            git clone https://github.com/JaySpiffy/IndexTTS-Workflow-Studio.git
            cd IndexTTS-Workflow-Studio
    ```
2.  **Create Conda Environment:**
    ```bash
    conda create -n index-tts-studio python=3.10
    conda activate index-tts-studio
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **FFmpeg:** You also need FFmpeg installed.
        *   Linux (Debian/Ubuntu): `sudo apt-get update && sudo apt-get install ffmpeg`
        *   macOS (using Homebrew): `brew install ffmpeg`
        *   Windows (using Chocolatey): `choco install ffmpeg` or download from the official FFmpeg website and add to your PATH.
    *   **Windows `pynini` Note:** If you encounter errors installing `pynini` on Windows (`Failed building wheel for pynini`), install it via conda *before* running `pip install -r requirements.txt`:
        ```bash
        conda install -c conda-forge pynini==2.1.5
        # Then run: pip install -r requirements.txt
        ```

4.  **Download Models:** Download the required model files (`bigvgan_discriminator.pth`, `bigvgan_generator.pth`, `bpe.model`, `dvae.pth`, `gpt.pth`, `unigram_12000.vocab`) from the links in the "Model Download" section above and place them into the `checkpoints/` directory within this project. You can use `wget` or `huggingface-cli`:
    *   Using `huggingface-cli`:
        ```bash
        # Recommended for China users: export HF_ENDPOINT="https://hf-mirror.com"
        huggingface-cli download IndexTeam/Index-TTS \
          bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
          --local-dir checkpoints --local-dir-use-symlinks False
        ```
    *   Using `wget`:
        ```bash
        wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_discriminator.pth -P checkpoints
        wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_generator.pth -P checkpoints
        wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bpe.model -P checkpoints
        wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/dvae.pth -P checkpoints
        wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/gpt.pth -P checkpoints
        wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/unigram_12000.vocab -P checkpoints
        ```

### Running the Web Demo (Workflow Studio)

```bash
python webui.py
```
Open your browser and visit `http://127.0.0.1:7860` (or the URL provided in the terminal) to access the Gradio interface.

### Running the API Server

```bash
# Ensure you are in the project root directory with the conda environment activated
uvicorn tts_api:app --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`. See `tts_api.py` for endpoint details (e.g., `/synthesize`).

### Basic Inference (Sample Code)
```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
voice="reference_voice.wav"
text="大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, output_path)
```

## Dialogue Generation Prompt

For those interested in the methodology, the detailed system prompt used to guide AI (like Cline) in generating the formatted dialogue scripts for the TTS engine can be found in the file [`DIALOGUE_GENERATION_PROMPT.md`](DIALOGUE_GENERATION_PROMPT.md).

## Acknowledge (Core Dependencies)
1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)

## 📚 Citation (Original IndexTTS Paper)

🌟 If you find our work helpful, please leave us a star and cite our paper.

```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025}
}
```
