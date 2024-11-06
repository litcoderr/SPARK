mkdir ~/dataset

git config --global user.email  "litcoderr@gmail.com"
git config --global user.name "Youngchae Chee"

export LD_LIBRARY_PATH="/usr/lib64-nvidia"

ANACONDA_VERSION="Anaconda3-2023.07-1-Linux-x86_64.sh"
curl -O https://repo.anaconda.com/archive/$ANACONDA_VERSION
bash $ANACONDA_VERSION -b -p $HOME/anaconda3
export PATH="$HOME/anaconda3/bin:$PATH"
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source $HOME/anaconda3/bin/activate
conda init --all --dry-run --force
rm $ANACONDA_VERSION

pip install datasets
pip install decord