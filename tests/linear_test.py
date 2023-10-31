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

class LinearTest_0:
    def set_configs(self, paddle):
        self.shape = [S_value, H_value]
        self.dtype = "float32"
        self.inputs = {
            "x": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
            "weight": paddle.randn([H_value,H_value], dtype=self.dtype),
            "bias": paddle.randn([S_value, H_value], dtype=self.dtype)
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

class LinearTest_1:
    def set_configs(self, paddle):
        self.shape = [S_value, H_value]
        self.dtype = "float32"
        self.inputs = {
            "x": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn([S_value, 4*H_value], dtype=self.dtype),
            "weight": paddle.randn([H_value,4*H_value], dtype=self.dtype),
            "bias": paddle.randn([S_value, 4*H_value], dtype=self.dtype)
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

class LinearTest_2:
    def set_configs(self, paddle):
        self.shape = [S_value, 4*H_value]
        self.dtype = "float32"
        self.inputs = {
            "x": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn([S_value, H_value], dtype=self.dtype),
            "weight": paddle.randn([4*H_value,H_value], dtype=self.dtype),
            "bias": paddle.randn([S_value, H_value], dtype=self.dtype)
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
    op_acc_stable_run(LinearTest_0)
    op_acc_stable_run(LinearTest_1)
    op_acc_stable_run(LinearTest_2)

