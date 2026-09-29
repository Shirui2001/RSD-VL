import argparse
import dataclasses
import json
from pathlib import Path
import time

import torch
from road_train import load_clip, sha256_file
from road_test import get_checkpoint_settings, load_weights, score_one_image


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--weights_dir',required=True)
    a.add_argument('--clip_checkpoint',required=True)
    a.add_argument('--output',required=True)
    a.add_argument('--warmup',type=int,default=20)
    a.add_argument('--repeats',type=int,default=100)
    args=a.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for useful throughput comparison')
    wd=Path(args.weights_dir)
    meta,cfg=get_checkpoint_settings(wd)
    if sha256_file(args.clip_checkpoint)!=meta['clip_sha256']:
        raise ValueError('Base CLIP checkpoint mismatch')
    import os
    os.environ['RSD_PROMPT_SETTING']=meta['prompt_setting']
    device=torch.device('cuda')
    model=load_clip(args.clip_checkpoint,int(meta['img_size']),device)
    load_weights(model,wd,meta)
    from forward_utils import get_adapted_multi_text_embeddings
    with torch.no_grad():
        normal,abnormal=get_adapted_multi_text_embeddings(model,'RoadAnomaly',device,requires_grad=False)
    inp=torch.randn(1,3,int(meta['img_size']),int(meta['img_size']),device=device)
    def run():
        return score_one_image(model,inp,normal,abnormal,cfg,
                               feature_space=meta['feature_space'])
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for _ in range(args.repeats):
        run()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    results={'params_m':sum(p.numel() for p in model.parameters())/1e6,
             'adapter_trainable_m':sum(p.numel() for p in model.image_adapter.parameters())/1e6 +
                                   sum(p.numel() for p in model.text_adapter.parameters())/1e6,
             'fps_batch1':args.repeats/elapsed,
             'gpu_peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,
             'gpu_device':torch.cuda.get_device_name(),
             'input_size':int(meta['img_size']),'batch_size':1,'dtype':'float32',
             'text_embeds_precomputed':True,'no_preprocessing_or_postprocessing':True,
             'warmup':args.warmup,'repeats':args.repeats,
             'note':'NEW benchmark only; not equivalent to vehicle end-to-end latency or Table14 baselines'}
    Path(args.output).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
