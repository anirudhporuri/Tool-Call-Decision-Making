from train_launcher_utils import apply_mode_defaults, build_train_argv, parse_train_launcher_args

def main(argv=None):
    args = parse_train_launcher_args(
        argv,
        description="Friendly launcher for LoRA SFT training.",
        dataset_config="train_sft",
        learning_rate=2e-4,
        warmup_steps=50,
    )
    apply_mode_defaults(args)
    from train_sft_lora import main as train_main

    train_main(build_train_argv(args))

if __name__ == "__main__":
    main()
