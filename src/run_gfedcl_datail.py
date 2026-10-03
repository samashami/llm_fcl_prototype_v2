#!/usr/bin/env python3
"""Run official GFedCL on the paper's frozen controlled Data-IL stream."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import types

import numpy as np
import torch

from src.external.gfedcl_datail import make_upstream_loader_factory

UPSTREAM_URL = "https://