#!/usr/bin/env python3

import itertools


def gen_addrs(base: str, bound: str, step: str, debug: bool = False, fname: str = None):
    """Generate hex values from a base to a bound, using a step.
    Inputs should be in hex
    """
    if debug:
        print(int(base, 16), int(step, 16), int(bound, 16))

    info = f"Writing {(int(bound, 16)-int(base, 16)) // int(step, 16)} values"
    if fname is not None:
        info += f" to {fname}"
    print(info)

    if fname is not None:
        out_f = open(fname, "a+")
    counter = itertools.count(int(base, 16), int(step, 16))
    while (val := next(counter)) < int(bound, 16):
        if debug:
            print(val, hex(val))
        elif fname is None:
            print(hex(val))

        if fname is not None:
            out_f.write(f"{hex(val)}\n")

    if fname is not None:
        out_f.close()


if __name__ == "__main__":
    gen_addrs("0x100000000", "0x380000000", "0x8000000", fname="addrs.dat")
