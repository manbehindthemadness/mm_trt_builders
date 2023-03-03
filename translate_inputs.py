"""
--optShapes=input_1:1x3x720x1280,input_2:1x1x720x1280

r1i:1,16,187,333 r2i:1,20,94,167 r3i:1,40,47,84 r4i:1,64,24,42
"""
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-shapes', type=str, required=True)
    parser.add_argument('--size', type=str, required=True, nargs=2)
    args = parser.parse_args()
    return args


def _main():
    """
    Do ittttt.
    """
    a = _parse_args()
    # shapes = f"src:1x3x{a.size[1]}x{a.size[0]}"
    shapes = ''
    shapes += a.onnx_shapes.replace('(', '').replace(')', '').replace(',', 'x').replace(' ', ',').strip()[:-1]
    print(shapes)
    return shapes  # src:1x3x1440x2560,r1i:1x16x187x333,r2i:1x20x94x167,r3i:1x40x47x84,r4i:1x64x24x42


if __name__ == '__main__':
    _main()
