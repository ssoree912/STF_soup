# python uwf_soup_stg_nf.py \
# --reference_args experiments/ShanghaiTech/prune_mag0p025_rnd0p025_ep1/seed_100/rand_100/pruning_soup_run1/args.json \
# --checkpoints experiments/ShanghaiTech/prune_mag0p025_rnd0p025_ep1/seed_100/rand_100/pruning_soup_run1/checkpoint_best.pth.tar \
# 							  experiments/ShanghaiTech/prune_mag0p025_rnd0p025_ep1/seed_100/rand_101/pruning_soup_run2/checkpoint_best.pth.tar \
# 							  experiments/ShanghaiTech/prune_mag0p025_rnd0p025_ep1/seed_100/rand_102/pruning_soup_run3/checkpoint_best.pth.tar \
#  --output results/nf_soup/prunesoup_model.pth.tar \
#   --device cuda:0 \
#   --max_batches 400 \
#   --subsample_ratio 0.5 \
#   --uw_var_unbiased \
#   --uw_shrink_alpha 0.7 \
#   --uw_gamma 0.5 \
#   --uw_qclip 98.5 \
#   --uw_wmin 0.2 \
#   --uw_wmax 5.0 \
#   --fisher_mix_eta 0.5 --sens_fisher_gamma 0.5 \
#   --sens_fisher_qclip 99.9 \
#   --actnorm_after_steps 500 \
#   --n_weightings 20 \
#   --fisher_floor 1e-8 \
#   --no_normalize_fishers \
#   --no_favor_target \
#   --log_level DEBUG \
#   --save_results_json \
#    --sens_regex "actnorm|log_scale|log_s" a

CUDA_VISIBLE_DEVICES=1 python uwf_soup_stg_nf.py \
--reference_args experiments/ShanghaiTech/baseline/seed_100/rand_none/baseline_soup_run1/args.json \
--checkpoints experiments/ShanghaiTech/baseline/seed_100/rand_none/baseline_soup_run1/checkpoint_best.pth.tar \
							  experiments/ShanghaiTech/baseline/seed_101/rand_none/baseline_soup_run2/checkpoint_best.pth.tar \
							  experiments/ShanghaiTech/baseline/seed_102/rand_none/baseline_soup_run3/checkpoint_best.pth.tar \
  --output results/nf_soup/baseline/baseline_soup_model1.pth.tar \
  --device cuda:0 \
  --max_batches 600 \
  --subsample_ratio 1.0 \
  --uw_var_unbiased \
  --uw_shrink_alpha 0.5 \
  --uw_gamma 0.5 \
  --uw_qclip 98.5 \
  --uw_wmin 0.5 \
  --uw_wmax 3.0 \
  --fisher_mix_eta 0.2 \
  --sens_regex "actnorm|log_scale|log_s" \
  --sens_fisher_gamma 0.5 \
  --sens_fisher_qclip 99.9 \
  --actnorm_after_steps 600 \
  --n_weightings 80 \
  --fisher_floor 1e-8 \
  --no_normalize_fishers \
  --log_level DEBUG \
  --save_results_json
