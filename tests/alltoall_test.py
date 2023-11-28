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

class AlltoallTest:
    def __init__(self, x_shape, y_shape, dtype):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.dtype = dtype 

    def set_configs(self, paddle):
        self.tmp_cache_path = "."
        self.inputs = {
            "x": paddle.randn(self.x_shape, dtype=self.dtype),
            "y": paddle.randn(self.y_shape, dtype=self.dtype),
        }

    def run_paddle(self, paddle):
        x = self.inputs["x"]
        y = self.inputs["y"]
        group = paddle.distributed.new_group([8])
        paddle.distributed.alltoall(x, y, group)
        return y

    def run_torch(self, torch):
        x = self.inputs["x"]
        y = self.inputs["y"]
        import os
        os.environ["MASTER_ADDR"] = '127.0.0.1' 
        os.environ["MASTER_PORT"] = '12345'
        torch.distributed.init_process_group(backend='nccl',rank=0,world_size=2)
        torch.distributed.all_to_all([y],[x])
        return y

    def check_diff(self, paddle, pd_ret, th_ret):
        assert len(pd_ret) == len(th_ret)
        for pd, th in zip(pd_ret, th_ret):
            check_tensor_diff(pd, th, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    op_acc_stable_run(AlltoallTest(x_shape=[8192, 31776], y_shape=[8192, 31776], dtype="bfloat16"))
