
from absl import flags
from absl import app
import sys
import os

from sotabencheval.language_modelling import WikiText103Evaluator, WikiText2Evaluator
import tensorflow as tf


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{dir_path}/tf")  # add tf. to dir path to import dynamic_eval
os.chdir(f"{dir_path}/tf")
from dynamiceval_tf_copy_for_sotabench import dynamic_eval, data_utils, FLAGS

def main(unused_argv=None):
    print("unused_argv", unused_argv)
    tf.logging.set_verbosity(tf.logging.INFO)
    # Get corpus info
    corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
    n_token = corpus_info["vocab_size"]
    cutoffs = corpus_info["cutoffs"][1:-1]
    tf.logging.info("n_token {}".format(n_token))

    evaluator = WikiText103Evaluator(
        model_name="Transformer-XL (RMS dynamic eval)",
        paper_arxiv_id="1904.08378",
        paper_pwc_id="dynamic-evaluation-of-transformer-language",
        #expected perplexity: 16.40
    ).eval(dynamic_eval(n_token, cutoffs, "/gpu:0")).print_results()

if __name__ == "__main__":
    import sys
    argv = f"""
    IGNORED-PROGNAME
        --data_dir={dir_path}/tf/pretrained_xl/tf_wt103/data/tfrecords
        --record_info_dir={dir_path}/tf/pretrained_xl/tf_wt103/data/tfrecords/
        --corpus_info_path={dir_path}/tf/pretrained_xl/tf_wt103/data/corpus-info.json
        --eval_ckpt_path={dir_path}/tf/pretrained_xl/tf_wt103/model/model.ckpt-0
        --model_dir=EXP-wt103
        --div_val=4
        --learning_rate=0.000002
        --decay_rate=0
        --epsilon=0.00001
        --rms=True
        --untie_r=True
        --proj_share_all_but_first=True
        --num_core_per_host=1
        --n_layer=18
        --d_model=1024
        --d_embed=1024
        --n_head=16
        --d_head=64
        --d_inner=4096
        --dropout=0.0
        --dropatt=0.0
        --tgt_len=128
        --mem_len=1600
        --clamp_len=1000
        --eval_split=test
        --same_length=True
    """.split()
    FLAGS(argv, known_only=True)
    assert FLAGS.data_dir == f"{dir_path}/tf/pretrained_xl/tf_wt103/data/tfrecords"
    main()
