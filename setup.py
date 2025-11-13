import os
import setuptools
import subprocess

from setuptools.command.build_py import build_py

third_party_include_dirs = (
    'third_party/cutlass/include',
    'third_party/cutlass/tools/util/include',
)

class CustomBuildPy(build_py):
    def run(self):
        # copy all third party include directories in advance
        src_dir = os.path.dirname(os.path.realpath(__file__))
        dst_dir = os.path.join(self.build_lib, 'qfactory/include')
        for include_dir in third_party_include_dirs:
            self.copy_tree(os.path.join(src_dir, include_dir), os.path.join(dst_dir, include_dir))

        build_py.run(self)

if __name__ == '__main__':
    # Get version from git commit
    try:
        revision = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except:
        revision = 'unknown-version'

    setuptools.setup(
        name='qfactory',
        version=f"0.0.1+{revision}",
        packages=[
            'qfactory',
            'qfactory/jit',
            'qfactory/kernels',
            'qfactory/linears',
        ],
        package_data={
            'qfactory': [
                'include/**/*'
            ]
        },
        cmdclass={
            'build_py': CustomBuildPy,
        }
    )