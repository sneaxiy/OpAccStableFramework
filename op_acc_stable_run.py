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

import inspect
import os
import pickle
import subprocess
import tempfile
from shlex import quote

import numpy as np


def convert_framework(
    inputs, tensor_to_numpy=True, dst_framework="torch", requires_grad=True
):
    assert tensor_to_numpy in [True, False], tensor_to_numpy
    assert dst_framework in ["paddle", "torch"], dst_framework
    assert requires_grad in [True, False], requires_grad

    if isinstance(inputs, (list, tuple)):
        outputs = [
            convert_framework(x, tensor_to_numpy, dst_framework, requires_grad)
            for x in inputs
        ]
        return type(inputs)(outputs)
    elif isinstance(inputs, dict):
        outputs = type(inputs)()
        for k, v in inputs.items():
            assert isinstance(k, str), "key {} should be str".format(k)
            outputs[k] = convert_framework(
                v,
                tensor_to_numpy,
                dst_framework,
                requires_grad=requires_grad and "_grad" not in k,
            )
        return outputs
    elif "paddle.Tensor" in str(type(inputs)):
        return inputs.detach().cpu().numpy()
    elif "torch.Tensor" in str(type(inputs)):
        import torch

        with torch.no_grad():
            inputs = inputs.contiguous()
            if inputs.dtype == torch.bfloat16:
                inputs = inputs.view(torch.int16)
            ret = inputs.detach().cpu().numpy()
            if inputs.dtype == torch.int16:
                ret = ret.view(np.uint16)
            return ret
    else:
        if not tensor_to_numpy and isinstance(inputs, np.ndarray):
            if dst_framework == "paddle":
                import paddle

                inputs = paddle.to_tensor(inputs)
                if requires_grad:
                    inputs.stop_gradient = False
                return inputs
            else:
                import torch

                with torch.no_grad():
                    inputs = np.ascontiguousarray(inputs)
                    if inputs.dtype == np.uint16:
                        inputs = inputs.view(np.int16)
                        inputs = torch.tensor(inputs, dtype=torch.int16).view(
                            torch.bfloat16
                        )
                    else:
                        inputs = torch.tensor(inputs)
                    inputs = inputs.cuda()
                    if requires_grad:
                        inputs.requires_grad_(True)
                    return inputs
        raise TypeError(str(type(inputs)))


def check_tensor_aadiff(x, y):
    import paddle

    assert x.dtype == y.dtype
    assert x.shape == y.shape
    if x.dtype in [paddle.bool, paddle.bfloat16]:
        x = x.astype(paddle.float32)
        y = y.astype(paddle.float32)
    assert paddle.max(paddle.abs(x - y)).numpy() == 0, "aadiff check failed"


def check_aadiff(x, y):
    if isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for xx, yy in zip(x, y):
            check_aadiff(xx, yy)
    else:
        check_tensor_aadiff(x, y)


def op_acc_stable_run(test_obj, stable_num=100):
    if inspect.isclass(test_obj):
        test_obj = test_obj()

    import paddle
    test_obj.init_params(paddle)
    test_obj.set_configs(paddle)

    src_path = os.path.abspath(inspect.getsourcefile(type(test_obj)))

    old_inputs = test_obj.inputs
    test_obj.inputs = convert_framework(old_inputs)

    envs = {
        "paddle": "",
        "torch": os.getenv("TORCH_VENV"),
    }

    ret = []
    with tempfile.TemporaryDirectory(dir="/home") as path:
        input_pickle_path = os.path.join(path, "inputs.bin")
        with open(input_pickle_path, "wb") as f:
            pickle.dump(test_obj, f)

        for framework, script in zip(
            ["paddle", "torch"], ["paddle_test.py", "torch_test.py"]
        ):
            test_path = os.path.join(path, script)
            output_pickle_path = os.path.join(path, f"{framework}_output.bin")
            with open(test_path, "w") as f:
                f.write(
                    f"""import {framework}
import pickle
from {os.path.basename(src_path).split('.')[0]} import *
from {os.path.basename(__file__).split('.')[0]} import convert_framework, check_aadiff

path = "{input_pickle_path}"
with open(path, 'rb') as f:
    test_obj = pickle.load(f)

stable_num = {stable_num} if "{framework}" == "paddle" else 1
prev_ret = None

old_inputs = test_obj.inputs
for i in range(stable_num):
    test_obj.inputs = convert_framework(
        old_inputs, tensor_to_numpy=False, dst_framework="{framework}")
    outputs = test_obj.run_{framework}({framework})

    if i == 0:
        prev_ret = outputs
        outputs = convert_framework(
            outputs, tensor_to_numpy=True, dst_framework="{framework}")
        with open("{output_pickle_path}", 'wb') as f:
            pickle.dump(outputs, f)
    else:
        with {framework}.no_grad():
            check_aadiff(prev_ret, outputs)
    print(i)

if stable_num > 1:
    print(f'AAdiff check passed after {stable_num} runs')
"""
                )

            import_paths = [src_path, "."]
            python_paths = []
            for p in import_paths:
                p = os.path.abspath(p)
                if os.path.isfile(p):
                    p = os.path.dirname(p)
                python_paths.append(p)

            venv = envs.get(framework)
            if venv:
                cmd = f'source "{envs.get(framework)}/bin/activate" && '
            else:
                cmd = ""

            cmd = (
                cmd
                + f'export PYTHONPATH=$PYTHONPATH:{":".join(python_paths)}'
                + f' && python "{test_path}"'
            )

            file_content = subprocess.check_output(["cat", test_path]).decode()

            assert (
                os.system(f"bash -c {quote(cmd)}") == 0
            ), f"{cmd} failed: \n\n{file_content}"
            with open(output_pickle_path, "rb") as f:
                outputs = pickle.load(f)
            ret.append(
                convert_framework(
                    outputs, tensor_to_numpy=False, dst_framework="paddle"
                )
            )

    test_obj.inputs = old_inputs

    import paddle

    with paddle.no_grad():
        test_obj.check_diff(paddle, ret[0], ret[1])
    print("Accuracy check passed")


def check_tensor_diff(x, y, *, atol, rtol, err_msg=""):
    assert x.dtype == y.dtype, f"{x.dtype} vs {y.dtype}, err_msg: {err_msg}"
    assert x.shape == y.shape, f"{x.shape} vs {y.shape}, err_msg: {err_msg}"
    import paddle

    with paddle.no_grad():
        if x.dtype == paddle.bfloat16:
            x = x.astype(paddle.float32)
            y = y.astype(paddle.float32)
        x = x.numpy()
        y = y.numpy()
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
