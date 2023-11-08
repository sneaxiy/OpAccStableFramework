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

class GeluTest:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def set_configs(self, paddle):
        self.inputs = {
            "x": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

    def run_paddle(self, paddle):
        x = self.inputs["x"]
        y = paddle.nn.functional.gelu(x)
        y.backward(self.inputs["y_grad"])
        return y, x.grad

    def run_torch(self, torch):
        x = self.inputs["x"]
        y = torch.nn.functional.gelu(x)
        y.backward(self.inputs["y_grad"])
        return y, x.grad

    def check_diff(self, paddle, pd_ret, th_ret):
        assert len(pd_ret) == len(th_ret)
        for pd, th in zip(pd_ret, th_ret):
            check_tensor_diff(pd, th, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    op_acc_stable_run(GeluTest(shape=[1, 12288], dtype="float32"))
    op_acc_stable_run(GeluTest(shape=[1, 12288], dtype="float16"))
    op_acc_stable_run(GeluTest(shape=[1, 12288], dtype="bfloat16"))
    op_acc_stable_run(GeluTest(shape=[1, 4096, 24576], dtype="float32"))
    op_acc_stable_run(GeluTest(shape=[1, 4096, 24576], dtype="float16"))
    op_acc_stable_run(GeluTest(shape=[1, 4096, 24576], dtype="bfloat16"))
