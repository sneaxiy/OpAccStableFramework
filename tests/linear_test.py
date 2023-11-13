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
B_value = 1
S_value = 4096
H_value = 12288

class LinearTest:
    def __init__(self, x_shape, y_grad_shape,weight_shape,bias_shape,dtype):
        self.x_shape = x_shape
        self.y_grad_shape = y_grad_shape
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.dtype = dtype

    def set_configs(self, paddle):
        self.tmp_cache_path = "."
        self.inputs = {
            "x": paddle.randn(self.x_shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.y_grad_shape, dtype=self.dtype),
            "weight": paddle.randn(self.weight_shape, dtype=self.dtype),
            "bias": paddle.randn(self.bias_shape, dtype=self.dtype)
        }


    def run_paddle(self, paddle):
        x = self.inputs["x"]
        weight = self.inputs["weight"]
        bias = self.inputs["bias"]
        y = paddle.nn.functional.linear(x,weight,bias)
        y.backward(self.inputs["y_grad"])
        return y, x.grad

    def run_torch(self, torch):
        x = self.inputs["x"]
        weight = self.inputs["weight"]
        weight=torch.transpose(weight ,dim0=0, dim1=1)
        bias = self.inputs["bias"]
        y = torch.nn.functional.linear(x,weight,bias)
        y.backward(self.inputs["y_grad"])
        return y, x.grad

    def check_diff(self, paddle, pd_ret, th_ret):
        assert len(pd_ret) == len(th_ret)
        for pd, th in zip(pd_ret, th_ret):
            check_tensor_diff(pd, th, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    op_acc_stable_run(LinearTest(x_shape=[S_value , H_value],y_grad_shape=[S_value , H_value],weight_shape=[H_value , H_value],bias_shape=[S_value , H_value], dtype="float32"))
    op_acc_stable_run(LinearTest(x_shape=[S_value , H_value],y_grad_shape=[S_value , H_value],weight_shape=[H_value , H_value],bias_shape=[S_value , H_value], dtype="float16"))
    op_acc_stable_run(LinearTest(x_shape=[S_value , H_value],y_grad_shape=[S_value , H_value],weight_shape=[H_value , H_value],bias_shape=[S_value , H_value], dtype="bfloat16"))
    
    op_acc_stable_run(LinearTest(x_shape=[S_value , H_value],y_grad_shape=[S_value, 4*H_value],weight_shape=[H_value,4*H_value],bias_shape=[S_value, 4*H_value], dtype="float32"))
    op_acc_stable_run(LinearTest(x_shape=[S_value , H_value],y_grad_shape=[S_value, 4*H_value],weight_shape=[H_value,4*H_value],bias_shape=[S_value, 4*H_value], dtype="float16"))
    op_acc_stable_run(LinearTest(x_shape=[S_value , H_value],y_grad_shape=[S_value, 4*H_value],weight_shape=[H_value,4*H_value],bias_shape=[S_value, 4*H_value], dtype="bfloat16"))

    op_acc_stable_run(LinearTest(x_shape=[S_value, 4*H_value],y_grad_shape=[S_value, H_value],weight_shape=[4*H_value,H_value],bias_shape=[S_value, H_value], dtype="float32"))
    op_acc_stable_run(LinearTest(x_shape=[S_value, 4*H_value],y_grad_shape=[S_value, H_value],weight_shape=[4*H_value,H_value],bias_shape=[S_value, H_value], dtype="float16"))
    op_acc_stable_run(LinearTest(x_shape=[S_value, 4*H_value],y_grad_shape=[S_value, H_value],weight_shape=[4*H_value,H_value],bias_shape=[S_value, H_value], dtype="bfloat16"))
