import os
import re
import copy
import uuid
import torch
import pickle
import hashlib
import logging
import subprocess
from types import NoneType
from torch.utils.cpp_extension import CUDA_HOME

from ..profile import profile_latency
from .runtime import Runtime, RuntimeCache

logger = logging.getLogger(__name__)

runtime_cache = RuntimeCache()

Python2CppTypes = { # (raw, actual)
    NoneType: ('void*', 'void*'),
    int: ('int', 'int'),
    torch.int32: ('void*', 'int*'),
    torch.float32: ('void*', 'float*'),
    torch.float16: ('void*', 'half_t*'),
    torch.bfloat16: ('void*', '__nv_bfloat16*'),
    torch.float8_e4m3fn: ('void*', '__nv_fp8_e4m3*'),
    torch.uint8: ('void*', 'uint8_t*'),
    torch.int8: ('void*', 'int8_t*'),
    torch.cuda.streams.Stream: ('void*', 'cudaStream_t'),
}

def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()[0:12]

def get_qfactory_version() -> str:
    include_dir = f'{os.path.dirname(os.path.abspath(__file__))}/../include/cuda'
    md5 = hashlib.md5()
    for root, _, files in os.walk(include_dir):
        for filename in filter(lambda x: x.endswith('.h'), sorted(files)):
            with open(os.path.join(root, filename), 'rb') as f:
                md5.update(f.read())
    return md5.hexdigest()[0:12]

def get_nvcc_compiler():
    nvcc_path = f'{CUDA_HOME}/bin/nvcc'
    version_pattern = re.compile(r'release (\d+\.\d+)')
    try:
        match = version_pattern.search(os.popen(f'{nvcc_path} --version').read())
        version = match.group(1)
        assert match is not None
    except:
        raise RuntimeError(f'Cannot get the version of NVCC compiler, CUDA_HOME={CUDA_HOME}')
    return nvcc_path, version

def get_cache_dir():
    if 'QFACTORY_CACHE_DIR' in os.environ:
        path = os.getenv('QFACTORY_CACHE_DIR')
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.expanduser('~') + '/.qfactory'

def write_file(tmp_dir: str, path: str, content: str):
    # Atomic write
    tmp_file_path = f'{tmp_dir}/file.tmp.{str(uuid.uuid4())}.{hash_to_hex(path)}'
    with open(tmp_file_path, 'w') as f:
        f.write(content)
    os.replace(tmp_file_path, path)



class JITCompiler:
    def __init__(self):
        self.tuned = {}

    @staticmethod
    def extract_arg_name_type(args):
        def to_type(arg):
            if isinstance(arg, torch.Tensor):
                return arg.dtype
            return type(arg)
        return tuple((name, to_type(arg)) for arg, name in args)

    def codegen(
        self,
        includes: tuple,
        template: str,
        keys: dict,
        args: tuple,
    ):
        code = '// QFactory Auto-generated Code\n\n'
        code += '\n'.join(f'#include "{inc}"' for inc in includes) + '\n\n'

        args_name_type = self.extract_arg_name_type(args)

        # launch function signature
        code += f'extern "C" void launch (\n'
        for name, t in args_name_type:
            raw_name = f'_{name}' if Python2CppTypes[t][0] != Python2CppTypes[t][1] else name
            code += f'\t{Python2CppTypes[t][0]} {raw_name},\n'
        code += '\tint& __return_code\n'
        code += ') {\n'

        # cast arguments
        for name, t in args_name_type:
            if Python2CppTypes[t][0] != Python2CppTypes[t][1]:
                raw_name = f'_{name}'
                code += f'\t{Python2CppTypes[t][1]} {name} = reinterpret_cast<{Python2CppTypes[t][1]}>({raw_name});\n'
        code += "\n"

        def cpp_parse(template: str, keys: dict) -> str:
            new_template = copy.deepcopy(template)
            for key, value in keys.items():
                new_template = new_template.replace(f'{{{key}}}', f'{value}')
            return new_template

        # function body
        body = cpp_parse(template, keys)
        code += '\n'.join([('\t' if line else '') + line for line in body.split('\n')])

        code += '}\n'

        logger.debug(f"Generated code:\n{code}\n")

        return code

    def compile(
        self,
        name: str,
        code: str,
        args: tuple,
    ) -> Runtime:
        args_name_type = self.extract_arg_name_type(args)

        arch = os.getenv('QFACTORY_ARCH', '80')

        nvcc_flags = [
            '-std=c++20', '-shared', '-O3', '--expt-relaxed-constexpr', '--resource-usage', '-Xptxas=-warn-lmem-usage',
            f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'
        ]
        cxx_flags = ['-fPIC', '-O3']
        flags = [*nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
        include_dirs = [
            f'{os.path.dirname(os.path.abspath(__file__))}/../include',
            f'{os.path.dirname(os.path.abspath(__file__))}/../include/third_party/cutlass/include',
            f'{os.path.dirname(os.path.abspath(__file__))}/../../third_party/cutlass/include',
        ]

        signature = f'{code}$${get_qfactory_version()}$${get_nvcc_compiler()}$${flags}$${include_dirs}$${arch}'
        runtime_name = f'kernel.{name}.{hash_to_hex(signature)}'
        runtime_path = f'{get_cache_dir()}/runtimes/{runtime_name}'
        tmp_dir = f'{get_cache_dir()}/tmp'
        
        global runtime_cache
        if runtime_cache[runtime_path] is not None:
            logger.debug(f"Using cached runtime for {runtime_name}")
            return runtime_cache[runtime_path]
        
        os.makedirs(runtime_path, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        args_path = f'{runtime_path}/kernel.args'
        src_path = f'{runtime_path}/kernel.cu'
        so_path = f'{runtime_path}/kernel.so'
        tmp_so_path = f'{tmp_dir}/nvcc.tmp.{str(uuid.uuid4())}.{hash_to_hex(so_path)}.so'

        write_file(tmp_dir, src_path, code)

        command = [get_nvcc_compiler()[0],
                src_path, '-o', tmp_so_path,
                *flags,
                *[f'-I{d}' for d in include_dirs]]
        logger.info(f'Compiling kernel {name} with command: {command}')
        return_code = subprocess.check_call(command)
        assert return_code == 0, f'Failed to compile {src_path}'

        pickle.dump(args_name_type, open(args_path, 'wb'))
        os.replace(tmp_so_path, so_path) # Atomic write

        runtime_cache[runtime_path] = Runtime(runtime_path)
        return runtime_cache[runtime_path]
    
    def compile_and_tune(
        self,
        name: str,
        includes: tuple,
        template: str,
        perf_keys: dict,
        keys: dict,
        space: tuple,
        args: tuple,
    ):
        keys = {k: keys[k] for k in sorted(keys)}
        identifier = (name, str(keys), str(perf_keys))
        if identifier in self.tuned:
            logger.debug(f"Using tuned result for {identifier}")
            return self.tuned[identifier]
        
        logger.info(f"Compiling and tuning {identifier}")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def run(config):
            full_keys = {**keys, **config}
            code = self.codegen(includes, template, full_keys, args)
            runtime = self.compile(name, code, args)
            return (runtime, config)
        
        all_kernels = []
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(run, config) for config in space]
            for future in as_completed(futures):
                all_kernels.append(future.result())

        run_args = tuple(arg for arg, _ in args)

        best_runtime, best_time, best_config = None, None, None
        if len(all_kernels) == 1:
            best_runtime, best_config = all_kernels[0]
            best_time = "[skip]"
        else:
            for runtime, config in all_kernels:
                ret_code = runtime.run(*run_args)
                if ret_code != 0:
                    logger.warning(f"Failed to run kernel {name} with config {config}: error code {ret_code}")
                    continue
                time = profile_latency(lambda : runtime.run(*run_args))
                if best_time is None or time < best_time:
                    best_runtime, best_time, best_config = runtime, time, config
                logger.debug(f"[Profile] Kernel {name} with config {config}: {time} us")
        
        assert best_runtime is not None, f"Failed to compile {identifier} with all configs"

        logger.info(f"Best config for {identifier}: {best_config} with latency {best_time} us")

        self.tuned[identifier] = best_runtime
        return best_runtime


jit = JITCompiler()