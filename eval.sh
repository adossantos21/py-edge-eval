# Copy this file to your working directory, make sure the environment where this package was installed is activated.
# On the following line, `cityscapes-raw-eval` was formerly `python scripts/cityscapes_raw.py`. For package-level install, it was made an executable.
cityscapes-raw-eval \
<path_to_cityscapes_root> \
<path_to_cityscapes_SBD_predictions> \
--output-path '<path_to_desired_output_folder>' \
--nonIS \
--pre-seal \
--remove-root \
--multi-label \
--max-dist 0.02 \
--thresholds 99 \
--nproc 8 \
--split 'val' \
--pred-suffix '.png' \
#--categories '1' \ # Uncomment to evaluate on a specific category

