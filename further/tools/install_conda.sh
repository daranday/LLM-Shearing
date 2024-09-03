wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /tmp
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

~/miniconda3/bin/conda init
conda create -n default python=3.10 -y