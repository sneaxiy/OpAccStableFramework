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


class AddTestCase:
    def __init__(self, x1_shape,x2_shape, dtype):
        self.x1_shape = x1_shape
        self.x2_shape = x2_shape
        self.dtype = dtype 
        
    def set_configs(self, paddle):
        self.tmp_cache_path = "."
        self.inputs = {
            "x1": paddle.randn(self.x1_shape, dtype=self.dtype),
            "x2": paddle.randn(self.x2_shape, dtype=self.dtype),
            "y_grad": paddle.randn(self.x1_shape, dtype=self.dtype),
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

if __name__ == "__main__":
    op_acc_stable_run(AddTestCase(x1_shape=[1], x2_shape=[1], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[1], x2_shape=[1], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[1], x2_shape=[1], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[1, 16, 4096, 128], x2_shape=[1, 16, 4096, 128], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[1, 16, 4096, 128], x2_shape=[1, 16, 4096, 128], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[1, 16, 4096, 128], x2_shape=[1, 16, 4096, 128], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[10944], x2_shape=[10944], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[10944], x2_shape=[10944], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[10944], x2_shape=[10944], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[2048,1,4096],x2_shape=[2048,1,4096], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[2048,1,4096],x2_shape=[2048,1,4096], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[2048,1,4096],x2_shape=[2048,1,4096], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[2048,1,4096],x2_shape=[4096], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[2048,1,4096],x2_shape=[4096], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[2048,1,4096],x2_shape=[4096], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape= [4096],x2_shape= [4096], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape= [4096],x2_shape= [4096], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape= [4096],x2_shape= [4096], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[50176,4096],x2_shape=[50176,4096], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[50176,4096],x2_shape=[50176,4096], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[50176,4096],x2_shape=[50176,4096], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[6144],x2_shape=[6144], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[6144],x2_shape=[6144], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[6144],x2_shape=[6144], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape= [1,8192,14,128], x2_shape=[1,8192,14,128], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape= [1,8192,14,128], x2_shape=[1,8192,14,128], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape= [1,8192,14,128], x2_shape=[1,8192,14,128], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape= [1,8192,14336], x2_shape=[1,8192,14336], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape= [1,8192,14336], x2_shape=[1,8192,14336], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape= [1,8192,14336], x2_shape=[1,8192,14336], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape= [12528,14336], x2_shape=[12528,14336], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape= [12528,14336], x2_shape=[12528,14336], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape= [12528,14336], x2_shape=[12528,14336], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape= [14336], x2_shape=[14336], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape= [14336], x2_shape=[14336], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape= [14336], x2_shape=[14336], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[14336,5376], x2_shape=[14336,5376], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[14336,5376], x2_shape=[14336,5376], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[14336,5376], x2_shape=[14336,5376], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[14336,9632], x2_shape=[14336,9632], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[14336,9632], x2_shape=[14336,9632], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[14336,9632], x2_shape=[14336,9632], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[1792,14336], x2_shape=[1792,14336], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[1792,14336], x2_shape=[1792,14336], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[1792,14336], x2_shape=[1792,14336], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape= [4816,14336], x2_shape=[4816,14336], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape= [4816,14336], x2_shape=[4816,14336], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape= [4816,14336], x2_shape=[4816,14336], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape=[5376], x2_shape=[5376], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape=[5376], x2_shape=[5376], dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape=[5376], x2_shape=[5376], dtype="bfloat16"))

    op_acc_stable_run(AddTestCase(x1_shape= [9632], x2_shape=[9632], dtype="float32"))
    op_acc_stable_run(AddTestCase(x1_shape= [9632], x2_shape=[9632],  dtype="float16"))
    op_acc_stable_run(AddTestCase(x1_shape= [9632], x2_shape=[9632],  dtype="bfloat16"))
