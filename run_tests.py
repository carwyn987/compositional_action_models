#!/usr/bin/env python3
"""Run every test under tests/: python run_tests.py [pytest args, e.g. -m unit]"""

import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-rs", *sys.argv[1:]]))
