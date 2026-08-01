"""Run every harness/test_*.py in a subprocess and summarize.

Usage: python harness/run_tests.py
Exit code is non-zero if any test fails.
"""

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    tests = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    results = []
    for t in tests:
        name = os.path.basename(t)
        t0 = time.time()
        p = subprocess.run([sys.executable, t], capture_output=True, text=True,
                           cwd=os.path.join(HERE, ".."))
        dt = time.time() - t0
        out = p.stdout.strip().splitlines()
        last = out[-1] if out else ""
        status = "PASS" if p.returncode == 0 else "FAIL"
        results.append((status, name, dt, last))
        print(f"{status}  {name:32s} {dt:7.1f}s  {last}")
        if p.returncode != 0:
            for line in (p.stdout.strip().splitlines() or [])[-12:]:
                print(f"      | {line}")
            err = p.stderr.strip().splitlines()
            for line in err[-8:]:
                print(f"      ! {line}")
    nfail = sum(1 for s, *_ in results if s == "FAIL")
    print(f"\n{len(results) - nfail}/{len(results)} passed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
