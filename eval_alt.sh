# On the following line, `cityscapes-raw-eval` was formerly `python scripts/cityscapes_raw.py`
python pyEdgeEval/scripts/evaluate/cityscapes_raw.py \
/home/robert.breslin/datasets/cityscapes \
/home/robert.breslin/alessandro/paper_2/mmsegmentation/SBD/ablation03/preds/edge4loss20/sbd_0 \
--output-path '/home/robert.breslin/alessandro/paper_2/mmsegmentation/SBD/ablation03/sbd_results' \
--categories '[1]' \
--nonIS \
--pre-seal \
--remove-root \
--max-dist 0.02 \
--thresholds 99 \
--nproc 24 \
--split 'val' \
--pred-suffix '_SBD.png' \
# --multi-label # Uncomment to evaluate SBD rather than Binary HED.