#!/usr/bin/env python3

import subprocess

import os

os.environ["USE_ADASWARM"] = "False"

exec(open('./examples/main.py').read())

os.environ["USE_ADASWARM"] = "True"

exec(open('./examples/main.py').read())