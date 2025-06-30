export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NTU-60 

python -m torch.distributed.launch --nproc_per_node=4 --master_port 10236 main_finetune.py \
    --config ./config/ntu60_xsub_joint/finetune_madiff_t120_layer8_decay.yaml \
    --output_dir <path-to-your-output-directory> \
    --log_dir <path-to-your-logging-directory> \
    --finetune <path-to-your-pretrained-checkpoint> \
    --dist_eval \
    --eval \
    --batch_size 16 \
    --accum_iter 1 \
    --epochs 100 \
    --warmup_epochs 5 \
    --model model.transformer_downstream.Transformer \
    --lr 3e-4 \
    --min_lr 1e-5
