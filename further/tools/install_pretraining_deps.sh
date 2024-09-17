# Install necessary packages
conda install -c conda-forge \
    htop \
    tmux \
    gh \
    tree \
    -y

# Install pip packages
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
MAKEFLAGS="-j$(nproc)" pip install flash-attn==1.0.3.post ipython
pip install -e ../..
pip install git+https://github.com/huggingface/transformers.git