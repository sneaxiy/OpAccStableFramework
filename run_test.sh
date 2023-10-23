# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

export TORCH_VENV=${TORCH_VENV:-"the_path_you_should_set"}

if ! axel -h >/dev/null; then
  apt-get install -y axel
fi

if [ ! -d "${TORCH_VENV}" ]; then 
  WHL_NAME="torch-2.1.0+cu118-cp38-cp38-linux_x86_64.whl"
  axel -a -n 16 -o "${WHL_NAME}" https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp38-cp38-linux_x86_64.whl
  python3.8 -m virtualenv "${TORCH_VENV}"
  (source "${TORCH_VENV}/bin/activate" && python -m pip install numpy "${WHL_NAME}")
  rm -rf "${WHL_NAME}"
fi

export PYTHONPATH="`dirname "$0"`:$PYTHONPATH"
python $@ 
