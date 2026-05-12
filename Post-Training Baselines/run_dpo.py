#!/usr/bin/env python3

from train_launcher_utils import apply_mode_defaults, build_train_argv, parse_train_launcher_args

def main(argv=None):
    args = parse_train_launcher_args(
        argv,
        description="Friendly launcher for LoRA DPO training.",
        dataset_config="train_pref",
        learning_rate=5e-6,
        warmup_steps=10,
        dpo=True,
    )
    apply_mode_defaults(args)
    from train_dpo_lora import main as train_main

    train_main(build_train_argv(args, dpo=True))

if __name__ == "__main__":
    main()
