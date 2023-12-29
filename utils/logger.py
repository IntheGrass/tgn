import logging
from pathlib import Path


def setup_logger():
    # 初始化logger，支持同时在命令行与文件中输出日志
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))  # 文件日志
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()  # 命令行日志
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
