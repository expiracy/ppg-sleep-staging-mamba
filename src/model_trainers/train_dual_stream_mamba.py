"""Train the dual-stream Mamba model (DS-M).

    python src/model_trainers/train_dual_stream_mamba.py --config configs/dsm_deep_256.yaml

The --use-tcn flags swap the Mamba context stack for dilated convolutions, an
ablation from the project report that the paper does not include.
"""

from model_trainers.trainer import build_arg_parser, load_config, run_training
from models.dual_stream_mamba import DualStreamMamba
from models.model_type import ModelType


def main():
    parser = build_arg_parser("Train the dual-stream Mamba (DS-M) model")
    parser.add_argument(
        "--use-tcn",
        action="store_true",
        help="Use TCN (dilated convolutions) instead of Mamba for the context stage",
    )
    parser.add_argument(
        "--tcn-kernel-size",
        type=int,
        default=None,
        help="Kernel size for TCN blocks (default: 7)",
    )
    parser.add_argument(
        "--tcn-dilations",
        type=str,
        default=None,
        help="Comma-separated dilation factors for TCN blocks (default: 1,2,4)",
    )
    args = parser.parse_args()
    config = load_config(args, parser)

    model_config = config.setdefault("model", {})
    if args.use_tcn:
        model_config["use_tcn"] = True
        print("Using TCN for the context stage (--use-tcn flag)")
    if args.tcn_kernel_size is not None:
        model_config["tcn_kernel_size"] = args.tcn_kernel_size
        print(f"TCN kernel size: {args.tcn_kernel_size}")
    if args.tcn_dilations is not None:
        dilations = [int(d.strip()) for d in args.tcn_dilations.split(",")]
        model_config["tcn_dilations"] = dilations
        print(f"TCN dilations: {dilations}")

    run_training(args, config, DualStreamMamba, ModelType.DUAL_STREAM_MAMBA)


if __name__ == "__main__":
    main()
