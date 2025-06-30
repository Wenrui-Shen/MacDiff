export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NTU-60 xsub
python -m torch.distributed.launch --nproc_per_node=4 --master_port 10235 main_linprobe.py \
    --config ./config/ntu60_xsub_joint/linprobe_madiff_t120_layer8.yaml \
    --output_dir <path-to-your-output-directory> \
    --log_dir <path-to-your-logging-directory> \
    --finetune <path-to-your-pretrained-checkpoint> \
    --dist_eval \
    --accum_iter 1 \
    --batch_size 64 \
    --epochs 100 \
    --model model.transformer_downstream.Transformer