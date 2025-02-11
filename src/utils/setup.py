from setuptools import setup, find_packages

setup(
    name="opencv_toolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.7.0",
        "scikit-learn>=1.0.0",
        "scikit-image>=0.19.0",
        "matplotlib>=3.5.0",
        "pillow>=9.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.65.0",
    ],
    author="OpenCV Toolkit Team",
    description="A comprehensive suite of image processing tools using OpenCV",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
