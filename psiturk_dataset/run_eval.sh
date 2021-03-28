#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting eval"
echo "hab sim: ${PYTHONPATH}"

python habitat_baselines/run.py --exp-config habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml --run-type eval
