#!/usr/bin/env python3
"""
简单的诊断脚本
"""

print("开始诊断...")

try:
    import sys
    print(f"Python版本: {sys.version}")
except Exception as e:
    print(f"导入sys失败: {e}")

try:
    import os
    print(f"当前目录: {os.getcwd()}")
except Exception as e:
    print(f"导入os失败: {e}")

try:
    import matplotlib
    print(f"matplotlib版本: {matplotlib.__version__}")
except Exception as e:
    print(f"导入matplotlib失败: {e}")

try:
    import numpy as np
    print(f"numpy版本: {np.__version__}")
except Exception as e:
    print(f"导入numpy失败: {e}")

try:
    import cv2
    print(f"opencv版本: {cv2.__version__}")
except Exception as e:
    print(f"导入cv2失败: {e}")

print("诊断完成")
