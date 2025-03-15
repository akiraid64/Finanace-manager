#!/usr/bin/env python3
"""
Build script for compiling the C++ extension module
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

def build_extension():
    """
    Build the C++ extension module using the appropriate build system
    """
    print("Building finance C++ extension module...")
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    cpp_dir = script_dir / "src" / "cpp"
    
    # Ensure the build directory exists
    build_dir = cpp_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Change to the build directory
    os.chdir(build_dir)
    
    # Run CMake
    try:
        print("Configuring with CMake...")
        subprocess.check_call(['cmake', '..'])
        
        print("Building with CMake...")
        subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'])
        
        # Copy the built extension to the src/python directory
        extension_suffix = get_extension_suffix()
        source_file = list(build_dir.glob(f"finance*{extension_suffix}"))[0]
        target_dir = script_dir / "src" / "python"
        target_dir.mkdir(exist_ok=True)
        target_file = target_dir / source_file.name
        
        print(f"Copying {source_file} to {target_file}")
        import shutil
        shutil.copy2(source_file, target_file)
        
        print("Build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during build: {e}")
        return False

def get_extension_suffix():
    """Get the appropriate suffix for the compiled extension module"""
    if platform.system() == "Windows":
        return ".pyd"
    else:
        return ".so"

def alternative_build():
    """
    Fallback build method if CMake is not available
    """
    print("Attempting alternative build method...")
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    cpp_dir = script_dir / "src" / "cpp"
    output_dir = script_dir / "src" / "python"
    output_dir.mkdir(exist_ok=True)
    
    os.chdir(cpp_dir)
    
    try:
        # Get the Python include path and extension suffix
        import distutils.sysconfig
        python_include = distutils.sysconfig.get_python_inc()
        
        # Try to get pybind11 include path
        try:
            import pybind11
            pybind_include = pybind11.get_include()
        except ImportError:
            print("Pybind11 not found. Please install it with pip install pybind11")
            return False
        
        # Build command
        if platform.system() == "Windows":
            # Windows build command
            build_cmd = [
                "cl", "/O2", "/Wall", "/EHsc", "/std:c++11", "/LD",
                f"/I{python_include}", f"/I{pybind_include}",
                "finance.cpp", "/link", "/DLL", f"/OUT:finance.pyd"
            ]
        else:
            # Unix build command
            build_cmd = [
                "c++", "-O3", "-Wall", "-shared", "-std=c++11", "-fPIC",
                f"-I{python_include}", f"-I{pybind_include}",
                "finance.cpp", "-o", f"finance{get_extension_suffix()}"
            ]
        
        print(f"Running build command: {' '.join(build_cmd)}")
        subprocess.check_call(build_cmd)
        
        # Copy the built extension to the output directory
        built_file = cpp_dir / f"finance{get_extension_suffix()}"
        target_file = output_dir / built_file.name
        
        import shutil
        shutil.copy2(built_file, target_file)
        
        print("Alternative build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Alternative build failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during alternative build: {e}")
        return False

if __name__ == "__main__":
    # Try the CMake build first
    if not build_extension():
        # If that fails, try the alternative build
        if not alternative_build():
            print("Both build methods failed. Please check your build environment.")
            sys.exit(1)
    
    sys.exit(0)
