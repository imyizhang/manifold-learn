#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools


with open('README.md', 'r', encoding='utf-8') as f:
    README = f.read()

setuptools.setup(
    name='manifold-learn',
    version='0.0.1',
    description='A lightweight toolkit making manifold learning much easier',
    author='Yi Zhang',
    author_email='yizhang.dev@gmail.com',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/imyizhang/manifold-learn',
    download_url='https://github.com/imyizhang/manifold-learn',
    packages=setuptools.find_packages(),
    keywords=[
        'pytorch',
        'manifold-learning',
        'dimension-reduction',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    license='MIT',
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'numba',
        'scikit-learn',
        'annoy',
        'torch',
    ],
)
