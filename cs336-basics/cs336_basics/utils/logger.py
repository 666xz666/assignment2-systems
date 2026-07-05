import logging
import os


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """
    初始化日志器：同时输出控制台 + 本地日志文件，带时间戳
    每次重启**追加日志**，不会覆盖旧日志
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    logger = logging.getLogger("cs336-basics")
    logger.setLevel(logging.INFO)
    # 避免重复添加处理器
    logger.handlers.clear()

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件持久化输出：mode='a' 追加写入，不覆盖历史
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
