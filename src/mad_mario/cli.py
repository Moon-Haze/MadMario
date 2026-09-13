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


def _use_headless_backend() -> None:
    """把 matplotlib 切到只画文件、不开窗口的 Agg。

    训练只把曲线写成 PNG，从不需要窗口；而 matplotlib 在有 DISPLAY 的机器上
    默认选 TkAgg，``plt.figure()`` 于是会建出 Tk 对象。这些对象只能在创建它们
    的主线程里销毁，可训练中那个录制最优模型的后台线程一样在分配内存、一样会
    触发分代 GC——一旦由它回收掉这些对象，tkinter 的 ``Variable.__del__`` 先抛
    "main thread is not in main loop"，紧接着 Tcl 以
    "Tcl_AsyncDelete: async handler deleted by the wrong thread" 终止整个进程。

    只能在 pyplot 首次导入之前调用；之后想让某个窗口消失也做不到了。所以训练
    与播放模块一律延迟到 main() 里按命令导入，播放那条路把后端留给 matplotlib
    自己挑——它可能真要弹窗。
    """
    import matplotlib

    matplotlib.use("Agg")


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
        # 先定后端，再让 trainer（→ metrics）把 pyplot 导进来
        _use_headless_backend()
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
