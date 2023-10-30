import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

th.manual_seed(1)
np.random.seed(1)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, model_base, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model_base.load_state_dict(
        dist_util.load_state_dict(args.model_base_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model_base.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
        model_base.convert_to_fp16()
    model.eval()
    model_base.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        if args.step:
            if args.combine:
                print("step+combine")
                sample_fn = (
                    diffusion.p_sample_loop_combine_step if not args.use_ddim else diffusion.ddim_sample_loop_combine_step
                )
                sample = sample_fn(
                    model,model_base,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    args.interval_t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            else:
                print("step+no combine")
                sample_fn = (
                    diffusion.p_sample_loop_step if not args.use_ddim else diffusion.ddim_sample_loop_step
                )
                sample = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    args.interval_t, #sample step interval 
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
        else:
            if args.combine:
                print("combine+nostep")
                sample_fn = (
                    diffusion.p_sample_loop_combine if not args.use_ddim else diffusion.ddim_sample_loop_combine
                )
                sample = sample_fn(
                    model,model_base,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            else:
                print("nocombine+nostep")
                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                sample = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 1, 3, 4, 2)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=100,
        use_ddim=False,
        interval_t=200,
        model_path="",
        model_base_path="",
        combine=False,
        step=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
