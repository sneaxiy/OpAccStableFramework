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

import numpy as np
from op_acc_stable_run import check_tensor_diff, op_acc_stable_run

class Tensor__div__Test:
    def __init__(self, x_shape, x_dtype, y_shape, y_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.y_shape = y_shape
        self.y_dtype = y_dtype


    def set_configs(self, paddle):
        self.tmp_cache_path = "."
        self.inputs = {
            "x": paddle.randn(self.x_shape, dtype=self.x_dtype) ,
            "y": paddle.randn(self.y_shape, dtype=self.y_dtype) ,
        }

    def run_paddle(self, paddle):
        x = self.inputs["x"]
        y = self.inputs["y"]
        y = paddle.Tensor.__div__(x, y)
        return y

    def run_torch(self, torch):
        x = self.inputs["x"]
        y = self.inputs["y"]
        y = torch.Tensor.__div__(x, y)
        return y

    def check_diff(self, paddle, pd_ret, th_ret):
        for pd, th in zip(pd_ret, th_ret):
            check_tensor_diff(pd, th, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    op_acc_stable_run(Tensor__div__Test(x_shape = [1, 8192], x_dtype = "float32", y_shape = [1], y_dtype = "float32"))
    op_acc_stable_run(Tensor__div__Test(x_shape = [1, 1024, 254208], x_dtype = "float32", y_shape = [1, 1024, 1], y_dtype = "float32"))
    