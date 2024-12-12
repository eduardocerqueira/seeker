#date: 2024-12-12T17:02:10Z
#url: https://api.github.com/gists/e0778620cf357a0ef5493469cdc457b3
#owner: https://api.github.com/users/qnzhou

#!/usr/bin/env python

import argparse

import lagrange   # pip install lagrange-open
import igl        # pip install libigl
import gpytoolbox # pip install gpytoolbox


def parse_args():
    parser = argparse.ArgumentParser(description="Check mesh loading")
    parser.add_argument("filename", type=str, help="Input mesh file")
    return parser.parse_args()


def main():
    args = parse_args()

    filename = args.filename

    lagrange_mesh = lagrange.io.load_mesh(filename)
    lagrange_V = lagrange_mesh.vertices
    lagrange_F = lagrange_mesh.facets

    igl_V, igl_F = igl.read_triangle_mesh(filename)

    gptb_V, gptb_F = gpytoolbox.read_mesh(filename)

    print(f"  Lagrange: #V={lagrange_V.shape[0]}")
    print(f"    LibIGL: #V={igl_V.shape[0]}")
    print(f"Gpytoolbox: #V={gptb_V.shape[0]}")


if __name__ == "__main__":
    main()
