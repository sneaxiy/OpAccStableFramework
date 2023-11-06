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


class AddTestCase0_FP32:
    def set_configs(self, paddle):
        self.shape = [1]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

    def run_paddle(self, paddle):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        y = paddle.add(x1, x2)
        y.backward(self.inputs["y_grad"])
        return y, x1.grad,x2.grad

    def run_torch(self, torch):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        y = torch.add(x1, x2)
        y.backward(self.inputs["y_grad"])
        return y, x1.grad,x2.grad

    def check_diff(self, paddle, pd_ret, th_ret):
        assert len(pd_ret) == len(th_ret)
        for pd, th in zip(pd_ret, th_ret):
            check_tensor_diff(pd, th, atol=1e-6, rtol=1e-6)

class AddTestCase0_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [1]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase1_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [1, 16, 4096, 128]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase1_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [1, 16, 4096, 128]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase2_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [10944]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase2_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [10944]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase3_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [2048,1,4096]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase3_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [2048,1,4096]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase4_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [2048,1,4096]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn([4096], dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase4_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [2048,1,4096]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn([4096], dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase5_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [4096]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase5_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [4096]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase6_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [50176,4096]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase6_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [50176,4096]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase7_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [6144]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase7_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [6144]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase8_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [1,8192,14,128]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase8_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [1,8192,14,128]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase9_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [1,8192,14336]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase9_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [1,8192,14336]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase10_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [12528,14336]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase10_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [12528,14336]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase11_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [14336]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase11_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [14336]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase12_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [14336,5376]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase12_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [14336,5376]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase13_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [14336,9632]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase13_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [14336,9632]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase14_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [1792,14336]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase14_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [1792,14336]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase15_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [4816,14336]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase15_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [4816,14336]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase16_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [5376]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase16_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [5376]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase17_FP32(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [9632]
        self.dtype = "float32"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }

class AddTestCase17_FP64(AddTestCase0_FP32):
    def set_configs(self, paddle):
        self.shape = [9632]
        self.dtype = "float64"
        self.inputs = {
            "x1": paddle.randn(self.shape, dtype=self.dtype),
            "x2": paddle.randn(self.shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.shape, dtype=self.dtype),
        }
                                                                                           
if __name__ == "__main__":
    # op_acc_stable_run(AddTestCase0_FP32)
    # op_acc_stable_run(AddTestCase0_FP64)
    # op_acc_stable_run(AddTestCase1_FP32)
    # op_acc_stable_run(AddTestCase1_FP64)
    # op_acc_stable_run(AddTestCase2_FP32)
    # op_acc_stable_run(AddTestCase2_FP64)
    # op_acc_stable_run(AddTestCase3_FP32)
    # op_acc_stable_run(AddTestCase3_FP64)
    # op_acc_stable_run(AddTestCase4_FP32)
    # op_acc_stable_run(AddTestCase4_FP64)
    op_acc_stable_run(AddTestCase5_FP32)
    op_acc_stable_run(AddTestCase5_FP64)
    op_acc_stable_run(AddTestCase6_FP32)
    op_acc_stable_run(AddTestCase6_FP64)
    op_acc_stable_run(AddTestCase7_FP32)
    op_acc_stable_run(AddTestCase7_FP64)
    op_acc_stable_run(AddTestCase8_FP32)
    op_acc_stable_run(AddTestCase8_FP64)
    op_acc_stable_run(AddTestCase9_FP32)
    op_acc_stable_run(AddTestCase9_FP64)
    op_acc_stable_run(AddTestCase10_FP32)
    op_acc_stable_run(AddTestCase10_FP64)
    op_acc_stable_run(AddTestCase11_FP32)
    op_acc_stable_run(AddTestCase11_FP64)
    op_acc_stable_run(AddTestCase12_FP32)
    op_acc_stable_run(AddTestCase12_FP64)
    op_acc_stable_run(AddTestCase13_FP32)
    op_acc_stable_run(AddTestCase13_FP64)
    op_acc_stable_run(AddTestCase14_FP32)
    op_acc_stable_run(AddTestCase14_FP64)
    op_acc_stable_run(AddTestCase15_FP32)
    op_acc_stable_run(AddTestCase15_FP64)
    op_acc_stable_run(AddTestCase16_FP32)
    op_acc_stable_run(AddTestCase16_FP64)
    op_acc_stable_run(AddTestCase17_FP32)
    op_acc_stable_run(AddTestCase17_FP64)
    

