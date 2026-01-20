#!/usr/bin/env python3
"""
print_reg_transforms.py

Scans a REG DICOM file or a directory of DICOM files and prints any transform/matrix-like
numeric elements found. If a 16-element vector is found it is printed as a 4x4 matrix.

Usage:
  python print_reg_transforms.py --file path/to/reg.dcm
  python print_reg_transforms.py --dir path/to/dicom_folder
  python print_reg_transforms.py --dir path/to/dicom_folder --to-ras

Note: This is a helper to extract matrices for manual testing in 3D Slicer. Matrices
printed are in the same numeric convention as found in the DICOM. If you need
LPS->RAS conversion, use --to-ras which applies a diagonal sign flip to X and Y.
"""

import argparse
import os
import re
import pydicom
import numpy as np


def try_parse_numeric_list(val):
    """Try to extract floats from a pydicom element value."""
    try:
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='ignore')
        s = str(val)
        parts = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', s)
        if parts:
            return [float(p) for p in parts]
    except Exception:
        return None
    return None


def find_matrix_like_elements(ds):
    """Iterate through dataset elements and collect numeric lists from elements whose
    names hint at transforms or matrices."""
    found = []
    for elem in ds.iterall():
        try:
            name = getattr(elem, 'name', None)
            if name is None:
                continue
            lname = name.lower()
            if any(k in lname for k in ('matrix','transform','registration','affine','parameters','matrix3x3')):
                vals = try_parse_numeric_list(elem.value)
                if vals:
                    found.append((name, vals))
        except Exception:
            continue
    return found


def format_matrix_4x4(M):
    lines = []
    for row in M:
        lines.append(' '.join(f'{v: .6f}' for v in row))
    return '\n'.join(lines)


def lps_to_ras_matrix(M):
    """Convert a 4x4 matrix from LPS to RAS coordinate convention by pre/post multiplying
    with diag(-1,-1,1,1). See DICOM vs RAS conventions. This flips X and Y axes.
    Note: This assumes the matrix is applied to homogeneous coordinates in row-major form.
    """
    S = np.diag([-1.0, -1.0, 1.0, 1.0])
    return S @ M @ S


def scan_file(fp, to_ras=False):
    try:
        ds = pydicom.dcmread(fp, force=True)
    except Exception as e:
        print(f"  -> Failed to read DICOM {fp}: {e}")
        return
    found = find_matrix_like_elements(ds)
    if not found:
        print('  -> No matrix-like numeric elements found in this file')
        return
    for name, nums in found:
        print(f"Element: {name} | numeric length: {len(nums)}")
        if len(nums) == 16:
            M = np.array(nums).reshape((4,4))
            print('Interpreting as 4x4 (row-major):')
            print(format_matrix_4x4(M))
            if to_ras:
                MR = lps_to_ras_matrix(M)
                print('\nConverted LPS->RAS (pre/post-multiplied):')
                print(format_matrix_4x4(MR))
        elif len(nums) == 12:
            M3x4 = np.array(nums).reshape((3,4))
            M = np.vstack([M3x4, [0.0, 0.0, 0.0, 1.0]])
            print('Interpreting as 3x4 (augmented to 4x4):')
            print(format_matrix_4x4(M))
            if to_ras:
                MR = lps_to_ras_matrix(M)
                print('\nConverted LPS->RAS:')
                print(format_matrix_4x4(MR))
        elif len(nums) == 9:
            R = np.array(nums).reshape((3,3))
            M = np.eye(4)
            M[:3,:3] = R
            print('Interpreting as 3x3 rotation (augmented to 4x4):')
            print(format_matrix_4x4(M))
            if to_ras:
                MR = lps_to_ras_matrix(M)
                print('\nConverted LPS->RAS:')
                print(format_matrix_4x4(MR))
        else:
            print('Raw numeric values (first 40):', nums[:40])


def scan_dir(path, to_ras=False):
    # Walk files, try reading DICOMs and scanning
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            print('\n--- FILE:', fp)
            scan_file(fp, to_ras=to_ras)


def main():
    p = argparse.ArgumentParser(description='Extract transform matrices from REG DICOM files')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--file', '-f', help='Path to a single REG DICOM file')
    g.add_argument('--dir', '-d', help='Path to a directory containing DICOM files to scan')
    p.add_argument('--to-ras', action='store_true', help='Also print matrix converted from LPS to RAS convention')
    args = p.parse_args()

    if args.file:
        print('Scanning file:', args.file)
        scan_file(args.file, to_ras=args.to_ras)
    else:
        print('Scanning directory:', args.dir)
        scan_dir(args.dir, to_ras=args.to_ras)


if __name__ == '__main__':
    main()
