IndexTTS Workflow Studio - Installation Guide

This guide will walk you through setting up the IndexTTS Workflow Studio on your system.

Author: James A Whittaker-Bent
Email: Whittakerbent@googlemail.com
1. Prerequisites

Before you begin, ensure you have the following installed:

    Git: For cloning the repository.

    Conda (Miniconda or Anaconda): Highly recommended for managing Python environments and dependencies. You can download it from Anaconda Distribution or Miniconda.

    Python: This project uses Python 3.10. We will create a Conda environment with this version.

    NVIDIA GPU (Recommended for Performance):

        For GPU acceleration, an NVIDIA GPU is required (e.g., your RTX 5070 Ti or older models like the RTX 3060 Ti).

        Ensure you have the latest NVIDIA drivers installed from the NVIDIA website.

        The project supports different CUDA versions depending on your GPU. We will cover PyTorch installation with CUDA below.

    FFmpeg: A system-level tool required for audio processing by libraries like pydub.

        Linux (Debian/Ubuntu):

        sudo apt-get update && sudo apt-get install ffmpeg

        macOS (using Homebrew):

        brew install ffmpeg

        Windows (using Chocolatey):

        choco install ffmpeg

        Alternatively, download from the official FFmpeg website, extract, and add the bin directory to your system's PATH environment variable.

2. Environment Setup

Follow these steps to set up your project environment:
Step 2.1: Clone the Repository

Open your terminal or command prompt and run:

git clone [https://github.com/JaySpiffy/IndexTTS-Workflow-Studio.git](https://github.com/JaySpiffy/IndexTTS-Workflow-Studio.git)
cd IndexTTS-Workflow-Studio

Step 2.2: Create and Activate Conda Environment

We will create a dedicated Conda environment to isolate project dependencies.

conda create -n index-tts-studio python=3.10
conda activate index-tts-studio

(Replace index-tts-studio with your preferred environment name if desired).
Step 2.3: Install PyTorch (Crucial Step!)

PyTorch is a core dependency. The version you install needs to be compatible with your NVIDIA GPU's CUDA capabilities (if you have one) or be a CPU-only version.
It is highly recommended to install PyTorch before other dependencies from requirements.txt.

    Go to the Official PyTorch Website: https://pytorch.org/get-started/locally/

    Use the Configurator: Select your OS (e.g., Windows), Package (Pip is generally recommended within a Conda environment for PyTorch compatibility), Compute Platform (CUDA version or CPU), and Python version (3.10).

    Run the Generated Command: The website will provide you with the exact command to install PyTorch.

Examples (Verify on PyTorch website for the latest commands):

    For Modern NVIDIA GPUs (e.g., RTX 30-series, RTX 40-series, RTX 50-series like your 5070 Ti) with CUDA 12.1 (or newer if available and supported):
    This project has been confirmed to work on an NVIDIA RTX 5070 Ti 16GB using a PyTorch build compatible with recent CUDA versions (e.g., CUDA 12.1 or newer). Always check the PyTorch website for the latest recommended command for your specific GPU and desired CUDA version.

    # Example for CUDA 12.1 - Verify on PyTorch.org! This is a common choice for recent GPUs.
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

    For Slightly Older NVIDIA GPUs that might prefer CUDA 11.8:

    # Example for CUDA 11.8 - Verify on PyTorch.org!
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

    For Older NVIDIA GPUs (or if CUDA 12.1/11.8 cause issues):
    If you have an older NVIDIA GPU, or if the newer CUDA versions (like 12.1 or 11.8) are not compatible or cause issues, you may need to install a PyTorch version built with an older CUDA toolkit (e.g., CUDA 11.7, 11.6, 11.3, 10.2).

        Check your GPU's CUDA compatibility: Refer to NVIDIA's official documentation for CUDA-Enabled GeForce and TITAN Products to see the maximum supported CUDA Toolkit version for your specific GPU model.

        Find a compatible PyTorch version: Visit the Previous PyTorch Versions page. Look for a PyTorch build that matches your OS, desired CUDA version (compatible with your GPU), and Python 3.10. The installation command will be provided there.
        Example for CUDA 11.7 (Verify on PyTorch.org for your specific needs!):

        # Example for CUDA 11.7 with pip - Always verify on PyTorch.org!
        # Replace <torch_version>, <torchvision_version>, and <torchaudio_version> with specific
        # compatible versions listed on the PyTorch previous versions page for CUDA 11.7.
        pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)

        (The versions 1.13.1, 0.14.1, 0.13.1 are examples for PyTorch with CUDA 11.7; always check the PyTorch website for the correct versions corresponding to your chosen CUDA toolkit and other dependencies.)

    For CPU-only installation (if you don't have an NVIDIA GPU or don't want GPU support):

    # Verify on PyTorch.org!
    pip3 install torch torchvision torchaudio

Important Notes on PyTorch Installation:

    The requirements.txt file in this repository may contain specific PyTorch versions. It's generally safer to install PyTorch manually using the commands from the official website first, and then install the rest of the requirements. If requirements.txt includes PyTorch, ensure it doesn't conflict with the version you need for your hardware. You can comment out the torch, torchaudio, and torchvision lines in requirements.txt if you install them manually.

    Your GPU drivers must be up-to-date and support the CUDA version you choose for PyTorch.

Step 2.4: Install Other Python Dependencies

Once PyTorch is installed and your Conda environment is active:

    (Windows Users - pynini): If you are on Windows, you might encounter issues installing pynini. It's recommended to install it via Conda before running the pip install -r requirements.txt command:

    conda install -c conda-forge pynini==2.1.5

    Install remaining dependencies:
    Now, install all other packages listed in the requirements.txt file:

    pip install -r requirements.txt

    This will install libraries like Gradio, Transformers, SpeechBrain, audio processing tools, etc.

3. Download Pre-trained Models

The core TTS models are large and are not included in this GitHub repository. You must download them separately.
Place the downloaded model files into a directory named checkpoints in the root of your IndexTTS-Workflow-Studio project folder.

Model Files to Download:

    bigvgan_discriminator.pth

    bigvgan_generator.pth

    bpe.model

    dvae.pth

    gpt.pth

    unigram_12000.vocab

(Refer to checkpoints/config.yaml if it lists other necessary model files)

Download Links:

    HuggingFace: Original IndexTTS Models (HuggingFace)

    ModelScope: Original IndexTTS Models (ModelScope) (Might be more accessible depending on your location)

Using huggingface-cli (Recommended):
Ensure you have huggingface-hub installed (pip install huggingface-hub).
For users in China, you might want to set the HuggingFace endpoint: export HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download IndexTeam/Index-TTS \
  bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints --local-dir-use-symlinks False

Using wget:

mkdir -p checkpoints
wget [https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_discriminator.pth](https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_discriminator.pth) -P checkpoints
wget [https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_generator.pth](https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_generator.pth) -P checkpoints
wget [https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bpe.model](https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bpe.model) -P checkpoints
wget [https://huggingface.co/IndexTeam/Index-TTS/resolve/main/dvae.pth](https://huggingface.co/IndexTeam/Index-TTS/resolve/main/dvae.pth) -P checkpoints
wget [https://huggingface.co/IndexTeam/Index-TTS/resolve/main/gpt.pth](https://huggingface.co/IndexTeam/Index-TTS/resolve/main/gpt.pth) -P checkpoints
wget [https://huggingface.co/IndexTeam/Index-TTS/resolve/main/unigram_12000.vocab](https://huggingface.co/IndexTeam/Index-TTS/resolve/main/unigram_12000.vocab) -P checkpoints

(Note: Ensure you have wget installed if using this method. On Windows, you might need to install it or use an alternative like curl or download manually via browser).
4. Running the Application

Once all dependencies and models are in place:
Step 4.1: Running the Web Demo (Workflow Studio)

Ensure your Conda environment (index-tts-studio) is activated. Then, from the project's root directory, run:

python webui.py

Open your web browser and navigate to the URL provided in the terminal (usually http://127.0.0.1:7860).
For detailed instructions on using the UI features, please see the Gradio UI Guide (you might need to create this or refer to project documentation).
Step 4.2: Running the API Server (Optional)

If you want to use the API:

# Ensure you are in the project root directory with the conda environment activated
uvicorn tts_api:app --host 0.0.0.0 --port 8000

The API will be available at http://localhost:8000. See tts_api.py for endpoint details (e.g., /synthesize).
5. Troubleshooting Common Issues

    ModuleNotFoundError: No module named 'X':

        Ensure your Conda environment is activated.

        Try pip install X.

        If it's a complex dependency, check the specific installation steps for that library.

    PyTorch CUDA Errors (CUDA_ERROR_NO_DEVICE, Torch not compiled with CUDA enabled, etc.):

        Verify you have a compatible NVIDIA GPU and the latest drivers.

        Ensure you installed the correct PyTorch build for your CUDA version (see Step 2.3).

        Check that CUDA is visible to PyTorch by running this Python script:

        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Current CUDA device index: {torch.cuda.current_device()}")
            print(f"Device name for current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            # If you want to list all available CUDA devices:
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. PyTorch will run on CPU.")

    webui.py fails to start or exits immediately:

        This could be due to a Python error in one of the project files. Check the terminal output carefully for any error messages when you run python webui.py.

        One issue that can cause this is a SyntaxError in constants.py, particularly around type definitions (e.g., for a type alias like ReviewYield). Ensure the syntax in this file is correct. If you encounter such an error, you might need to compare it with related definitions in other files (like webui.py) or ensure it matches the intended Python syntax for type hints.

    FFmpeg not found:

        Ensure FFmpeg is installed and its bin directory is added to your system's PATH environment variable.

        You can test by typing ffmpeg -version in your terminal.

    Model Download Issues:

        Check your internet connection.

        Try the alternative download link (ModelScope if HuggingFace is slow/blocked).

        Ensure you have enough disk space.

        Verify the checkpoints directory is created in the correct location (root of the project).

    pynini installation fails on Windows:

        Make sure you run conda install -c conda-forge pynini==2.1.5 before pip install -r requirements.txt within your activated Conda environment.

6. Further Assistance

If you encounter issues not covered here, please check the project's GitHub Issues page or consider reaching out to the author.