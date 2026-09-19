"""Train the single-stream Mamba model (SS-M).

    python src/model_trainers/train_single_stream_mamba.py --config configs/ssm_shallow_192.yaml

The config's model section sets the width and the number of Mamba blocks.
"""

from model_trainers.trainer import build_arg_parser, load_config, run_training
from models.model_type import ModelType
from models.single_stream_mamba import SingleStreamMamba


def main():
    parser = build_arg_parser("Train the single-stream Mamba (SS-M) model")
    args = parser.parse_args()
    config = load_config(args, parser)
    run_training(args, config, SingleStreamMamba, ModelType.SINGLE_STREAM_MAMBA)


if __name__ == "__main__":
    main()
