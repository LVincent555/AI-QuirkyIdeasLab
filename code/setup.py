# setup.py
from setuptools import setup, Extension
import pybind11

# 定义C++扩展模块
ext_modules = [
    Extension(
        'maigem_core', # 模块名
        ['maigem_core.cpp'], # 源文件
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17'], # 使用C++17标准和优化
    ),
]

setup(
    name='maigem_core',
    version='1.0',
    author='AI Assistant',
    description='MAI-GEM Core C++ extension',
    ext_modules=ext_modules,
)
# 本代码采用 GNU General Public License v3.0 (GPL 3.0) 开源协议。  
# 作者：vincent  
# 协议链接：https://www.gnu.org/licenses/gpl-3.0.html  