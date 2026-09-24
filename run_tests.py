"""Run every test under tests/: python run_tests.py [pytest args, e.g. -m unit]"""

import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:]))
