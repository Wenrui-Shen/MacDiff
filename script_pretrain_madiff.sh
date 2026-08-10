export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3 # change this and `nproc_per_node` accordingly

# change the paths below to your own

python -m torch.distributed.launch --nproc_per_node=4 --master_port 10234 main_pretrain.py \
    --config ./config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml \
    --ose_exemplar_indices <path-to-exemplar-indices.json> \
    --output_dir <path-to-your-output-directory> \
    --log_dir <path-to-your-logging-directory> \
    --batch_size 32 \
    --accum_iter 1 \
    --epochs 400 \
    --lr 1e-3 \
    --min_lr 1e-5 \
    --mask_ratio 0.9 \
    --model model.transformer_macdiff.Transformer
