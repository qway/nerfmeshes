#!/bin/bash
sudo apt-get install libffi-dev
curl https://pyenv.run | bash

# add to path
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# apply envs
source ~/.bashrc

pyenv install 3.7.7
pyenv global 3.7.7

# set sym link, the arguments may be different
ln -s ../drive/My\ Drive/data data

# install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# add poetry to path
echo 'export PATH="$HOME/.poetry/bin:$PATH"' >> ~/.bashrc

# apply envs
source ~/.bashrc

# Configure poetry to create virtual environments inside the project's root directory
poetry config virtualenvs.in-project true

# install dependencies
poetry install

# activate env
source .venv/bin/activate

