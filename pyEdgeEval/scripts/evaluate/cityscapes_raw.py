#!/usr/bin/env python3

from pyEdgeEval.helpers.evaluate_cityscapes import evaluate_cityscapes_raw

def main():
    evaluate_cityscapes_raw(gt_dir='gtEval')
if __name__ == "__main__":
    main()
