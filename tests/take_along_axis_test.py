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

class TakealongaxisTest:
    def __init__(self, arr_shape,arr_dtype, indices_shape,indices_dtype, axis):
        self.arr_shape = arr_shape 
        self.arr_dtype = arr_dtype 
        self.indices_shape = indices_shape 
        self.indices_dtype = indices_dtype 
        self.axis = axis 

    def set_configs(self, paddle):
        self.tmp_cache_path = "."
        self.inputs = {
            "arr": paddle.to_tensor(np.random.random(size=self.arr_shape).astype(self.arr_dtype) - 0.5) ,
            "indices": paddle.to_tensor(np.random.random(size=self.indices_shape).astype(self.indices_dtype)),
        }

    def run_paddle(self, paddle):
        arr = self.inputs["arr"]
        indices = self.inputs["indices"]
        y = paddle.take_along_axis(arr, indices, self.axis)
        return y

    def run_torch(self, torch):
        arr = self.inputs["arr"]
        indices = self.inputs["indices"]
        indices = indices.numpy().astype(self.indices_dtype)
        y = np.take_along_axis(arr, indices, self.axis)
        return y

    def check_diff(self, paddle, pd_ret, th_ret):
        assert len(pd_ret) == len(th_ret)
        for pd, th in zip(pd_ret, th_ret):
            check_tensor_diff(pd, th, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    op_acc_stable_run(TakealongaxisTest(arr_shape=[1, 1024, 254208], arr_dtype="float32", indices_shape=[1, 1024, 1], indices_dtype="int64" , axis=-1))
    