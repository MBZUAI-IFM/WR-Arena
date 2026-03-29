# SEA-RAFT Checkpoints

Download the pretrained checkpoint used for smoothness evaluation:

```bash
# Tartan-C-T-TSKH-spring540x960-M (recommended)
wget https://huggingface.co/datasets/memcpy/SEA-RAFT/resolve/main/Tartan-C-T-TSKH-spring540x960-M.pth \
    -O thirdparty/SEA-RAFT/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth
```

Or visit the [SEA-RAFT repository](https://github.com/princeton-vl/SEA-RAFT) for other checkpoint options.

Pass the downloaded checkpoint path to the eval script:
```bash
python smoothness_eval_scripts/compute_smoothness_scores.py \
    --raft_ckpt thirdparty/SEA-RAFT/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth \
    ...
```
