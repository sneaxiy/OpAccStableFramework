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


from op_acc_stable_run import check_tensor_diff, op_acc_stable_run

class CastTest:
    def __init__(self, x_shape, dtype):
        self.x_shape = x_shape
        self.dtype = dtype 

    def set_configs(self, paddle):
        self.tmp_cache_path = "."
        self.inputs = {
            "x": paddle.randn(self.x_shape) ,
        }

    def run_paddle(self, paddle):
        x = self.inputs["x"]
        y = paddle.cast(x, self.dtype)
        return y

    def run_torch(self, torch):
        def convert_dtype(dtype):
            ret = None
            if dtype == "float32":
                ret = torch.float32
            elif dtype == "float16":
                ret = torch.float16
            elif dtype == "bfloat16":
                ret = torch.bfloat16
            elif dtype == "int64":
                ret = torch.int64
            elif dtype == "uint16":
                ret = torch.bfloat16
            return ret
        x = self.inputs["x"]
        type = convert_dtype(self.dtype)
        y = x.to(type) 
        return y

    def check_diff(self, paddle, pd_ret, th_ret):
        assert len(pd_ret) == len(th_ret)
        for pd, th in zip(pd_ret, th_ret):
            check_tensor_diff(pd, th, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    op_acc_stable_run(CastTest(x_shape = [1, 1024, 254208], dtype ='float32'))
