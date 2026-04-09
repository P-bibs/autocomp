import pathlib
import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import os
import shutil
from datetime import datetime

from autocomp.common import logger, SOLS_DIR
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.backend.eval_backend import EvalBackend

KERNELBENCH_DIR = pathlib.Path("./KernelBench")


def _discover_cuda_devices() -> list[str]:
    """Return visible CUDA device ids for parallel candidate evaluation."""
    configured = os.environ.get("AUTOCOMP_CUDA_DEVICES")
    if configured:
        return [device.strip() for device in configured.split(",") if device.strip()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [device.strip() for device in visible.split(",") if device.strip()]

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ["0"]

    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return devices or ["0"]

class KBEvalBackend(EvalBackend):
    def preprocess_code_for_evaluation(self, prob: Prob, code_str: str) -> str:
        sol_files_dir = KERNELBENCH_DIR / "KernelBenchFunctional" / prob.prob_type.replace("kb-","")
        matches = list(sol_files_dir.glob(f"{prob.prob_id}_*.py"))
        if matches:
            lines = []
            # get all lines beween `# BEGIN EVAL UTILS` and `# END EVAL UTILS` in the eval file
            with open(matches[0], "r") as f:
                in_utils = False
                for line in f:
                    if line.strip() == "# BEGIN EVAL UTILS":
                        in_utils = True
                    elif line.strip() == "# END EVAL UTILS":
                        in_utils = False
                    elif in_utils:
                        lines.append(line)
            utils_code = "".join(lines)
            return utils_code + "\n\n" + code_str
        raise ValueError(f"No matching eval utils file found for prob {prob.prob_type} {prob.prob_id} in {sol_files_dir}")

    def get_backend_specific_rules(self) -> list[str]:
        return [
            "You're goal is to optimize the implemention of functional_model, which will be evaluated for correctness and performance.",
            "All generated code should be contained in a single Python file (inline CUDA code is allowed).",
            r'''
You can use the following Python code to build a CUDA extension for Torch:
```python
from torch.utils.cpp_extension import load_inline
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(...) {
    ...
}

void fused_op_forward(int blocks, int threads, ...) {
    fused_op_forward_kernel<<<blocks, threads>>>(...);
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(...);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "<docstring goes here>");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

fused_ext.fused_op(...)
```
''',
            # "When using torch.utils.cpp_extension load() or load_inline(), make sure to place C++ code in cpp_sources and CUDA code in cuda_sources.",
            "Do not use the `function` argument of load_inline(), make a PYBIND11 binding instead.",
            # "Don't do any work in the non-device CUDA and C++ code. All work (including grid/block size selection and tensor allocation) should be done in the Python code. The arguments passed to the inline module should be directly passed through to the CUDA kernel without modification.",
            "By the time optimization is complete, the code should not use built-in pytorch matmul or convolution functions. Use your own CUDA kernels instead",
            "Only function functional_model will be imported during evaluation. Feel free to define other variables, functions, or classes, but make sure they are used by functional_model.",
        ]

    def evaluate_code(
        self,
        prob: Prob,
        code_strs: list[str],
        simulator: str,
        candidates: list[CodeCandidate] | None = None,
    ) -> List[dict]:
        level_str = prob.prob_type.split("-")[1]
        ref_file = glob.glob(f"{KERNELBENCH_DIR}/KernelBench/{level_str}/{prob.prob_id}_*.py")[0]
        ref_file = os.path.abspath(ref_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = pathlib.Path(__file__).parent / "tmp_files" / f"kb_eval_{timestamp}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cuda_devices = _discover_cuda_devices()
        max_parallel = min(len(code_strs), len(cuda_devices)) if code_strs else 1

        def _evaluate_single(i: int, code_str: str) -> dict:
            test_file = tmp_dir / f"code_{i}.py"
            transformed_code = self.preprocess_code_for_evaluation(prob, code_str)
            test_file.write_text(transformed_code)
            assigned_device = cuda_devices[i % len(cuda_devices)]

            cmd = [
                "uv",
                "run",
                "python",
                "scripts/run_and_check.py", 
                "ref_origin=local",
                f"ref_arch_src_path={str(ref_file)}",
                f"kernel_src_path={str(test_file)}",
                f"level={level_str[-1]}",
                f"problem_id={prob.prob_id}",
                "timeout=10",
                "check_kernel=False",
            ]
            child_env = os.environ.copy()
            child_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            child_env["CUDA_VISIBLE_DEVICES"] = assigned_device
            logger.info(
                "Running command on GPU %s: %s from cwd %s",
                assigned_device,
                " ".join(cmd),
                KERNELBENCH_DIR,
            )
            try:
                result = subprocess.run(
                    cmd,
                    cwd=KERNELBENCH_DIR,
                    env=child_env,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
            except Exception as e:
                logger.info(f"Error running command: {e}")
                return {"correct": False}
            stdout = result.stdout
            output_file = tmp_dir / f"output_{i}.txt"
            output_file.write_text(stdout)
            if " runtime_stats={'mean':" not in stdout:
                logger.info(f"Kernel did not pass correctness for code {i}")
                # print stdout but with tabs prepended
                for line in stdout.splitlines():
                    logger.info(f"\t{line}")
                return {"correct": False}
            else:
                latency = float(stdout.split(" runtime_stats={'mean': ")[-1].split(",")[0])
                plan_model = None
                code_model = None
                if candidates is not None and i < len(candidates):
                    plan_model = candidates[i].plan_gen_model
                    code_model = candidates[i].code_gen_model
                logger.info(
                    "Kernel passed correctness for code %d, plan_model: %s, code_model: %s, latency: %s",
                    i,
                    plan_model or "unknown",
                    code_model or "unknown",
                    latency,
                )
                return {"correct": True, "latency": latency}

        if max_parallel <= 1:
            return [_evaluate_single(i, code_str) for i, code_str in enumerate(code_strs)]

        logger.info(
            "Evaluating %d candidates in parallel across %d GPU(s): %s",
            len(code_strs),
            max_parallel,
            ",".join(cuda_devices[:max_parallel]),
        )
        results = [None] * len(code_strs)
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(_evaluate_single, i, code_str): i
                for i, code_str in enumerate(code_strs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

def main():
    prob_type = "kb-level1"
    prob_id = 1
    prob = Prob(prob_type, prob_id)
    files = glob.glob(str(SOLS_DIR / prob_type / f"{prob_id}_*.py"))
    code_strs = [pathlib.Path(file).read_text() for file in files]
    stats = KBEvalBackend().evaluate_code(prob, code_strs, "kernelbench")
    print(stats)

if __name__ == "__main__":
    main()
