import os

root_dir = os.path.expanduser("~")
data_dir = "summary/data/"
source_dir = "summary/pointer_summarizer/"
train_data_path = os.path.join(root_dir, data_dir, "finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, data_dir, "finished_files/val.bin")
decode_data_path = os.path.join(root_dir, data_dir, "finished_files/test.bin")
vocab_path = os.path.join(root_dir, data_dir, "finished_files/vocab")
log_root = os.path.join(root_dir, source_dir, 'log/')
model_file_path="/home/ta/summary/pointer_summarizer/log/train_1551439117/model/model_20000_1551488065"

# Hyperparameters
do_lower_case = True
hidden_dim= 256
emb_dim= 128

# bert
bert_dim=768
bert_model="bert-base-uncased"

batch_size=8
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = True
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15
