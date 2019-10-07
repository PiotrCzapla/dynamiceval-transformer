#!/usr/bin/env bash -x
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}
REPO="$( cd "$(dirname "$0")" ; pwd -P )"
$PYTHON -m pip install 'tensorflow-gpu==1.14'

apt install -y  python2.7 python-pip
apt install -y wget
# to run legacy scripts on dynamic eval
python2.7 -m pip install 'tensorflow-gpu==1.14'
cd $REPO/tf

# Data
DATA_ROOT=$REPO/tf
mkdir -p ~/.cache/torch/pretrained_xl

ln -fs ~/.cache/torch/pretrained_xl ${DATA_ROOT}/pretrained_xl
DATA_DIR=${DATA_ROOT}/pretrained_xl/tf_wt103/data
MODEL_DIR=${DATA_ROOT}/pretrained_xl/tf_wt103/model

URL=http://curtis.ml.cmu.edu/datasets/pretrained_xl
function download () {
  fileurl=${1}
  filename=${fileurl##*/}
  if [ ! -f ${filename} ]; then
    echo ">>> Download '${filename}' from '${fileurl}'."
    wget --quiet ${fileurl}
  else
    echo "*** File '${filename}' exists. Skip."
  fi
}

# wt103
if [ ! -d $DATA_DIR/cache.pkl ]; then
    cd $DATA_ROOT/pretrained_xl
    mkdir -p tf_wt103 && cd tf_wt103

    mkdir -p data && cd data
    download ${URL}/tf_wt103/data/cache.pkl
    download ${URL}/tf_wt103/data/corpus-info.json
    cd $DATA_ROOT
fi

if [ ! -d $MODEL_DIR/checkpoint ]; then
    mkdir -p $MODEL_DIR
    cd $MODEL_DIR
    download ${URL}/tf_wt103/model/checkpoint
    download ${URL}/tf_wt103/model/model.ckpt-0.data-00000-of-00001
    download ${URL}/tf_wt103/model/model.ckpt-0.index
    download ${URL}/tf_wt103/model/model.ckpt-0.meta
    cd $DATA_ROOT
fi

TEST_TGT_LEN=128

echo 'Preprocess test set...'
if  [ ! -d $DATA_DIR/tfrecords ] ; then
    python2.7 data_utils.py \
        --data_dir=${DATA_DIR}/ \
        --dataset=wt103 \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=1 \
        --num_passes=1 \
        --use_tpu=False

    #TODO Why this is being exectued second time with different bsz ?
    python2.7 data_utils.py \
        --data_dir=${DATA_DIR}/ \
        --dataset=wt103 \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=0 \
        --num_passes=1 \
        --use_tpu=False 
fi

SOTABENCH=/workspace/
if [ -d $SOTABENCH ]; then 
    echo 'Running temporary setup scripts'
    mkdir -p $SOTABENCH
    cd $SOTABENCH
    git clone https://github.com/PiotrCzapla/sotabench-eval.git
    cd sotabench-eval
    git pull
    pip install -e .
fi