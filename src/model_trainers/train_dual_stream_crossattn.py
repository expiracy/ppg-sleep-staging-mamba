"""Train the dual-stream cross-attention baseline (DS-CA).

    python src/model_trainers/train_dual_stream_crossattn.py --config configs/dsca.yaml

The second stream is a noise-corrupted copy of the PPG, generated on the fly, so this
still trains from the single extracted PPG signal.
"""

from model_trainers.trainer import build_arg_parser, load_config, run_training
from models.model_type import ModelType
from models.ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention


def main():
    parser = build_arg_parser("Train the dual-stream cross-attention (DS-CA) baseline")
    args = parser.parse_args()
    config = load_config(args, parser)
    run_training(args, config, PPGUnfilteredCrossAttention, ModelType.PPG_UNFILTERED)


if __name__ == "__main__":
    main()
