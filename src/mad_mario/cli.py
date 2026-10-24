import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# 限制 BLAS 线程数防止多进程环境下的线程爆炸（每个子进程 × 自动线程数 >= ULIMIT）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from mad_mario.config import (
    _apply_config_defaults,
    _extract_config_path,
    build_parser,
    config_from_play_args,
    config_from_train_args,
    load_config_file,
)
from mad_mario.logging.setup import setup_logging


def main():
    # 1. 提前扫描 --config（不经过 argparse，因为需要在解析前获得配置）
    config_path = _extract_config_path()
    config = {}
    if config_path:
        config = load_config_file(config_path)

    # 2. 构建解析器
    parser = build_parser()

    # 3. 将配置项作为默认值注入所有子解析器
    if config:
        _apply_config_defaults(parser, config)

    # 4. 正常解析——CLI 参数自然覆盖 config 中的对应值
    args = parser.parse_args()

    if args.command == "train":
        # matplotlib 的 Agg 后端由 metrics 模块自己保证（训练只写 PNG 不开
        # 窗口），这里不再背负"先定后端再导入"的顺序责任
        from mad_mario.training.trainer import train

        config = config_from_train_args(args)
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        log_path = setup_logging(level=log_level, log_dir=config.artifacts.save_root / "logs")
        train(config, log_path=log_path)
    elif args.command == "play":
        from mad_mario.playback.runner import play

        setup_logging()
        play(config_from_play_args(args))
    else:
        parser.error(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
