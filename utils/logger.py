# coding=utf-8
import os
import logging


def logger_init():
    # Log等级总开关
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建log目录
    if not os.path.exists('./logfiles'):
        os.mkdir('./logfiles')

    # 创建一个handler，用于写入日志文件
    logfile = './logfiles/robot-ctl-python.log'

    # 以append模式打开日志文件
    fh = logging.FileHandler(logfile, mode='a')

    # 输出到file的log等级的开关
    fh.setLevel(logging.INFO)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()

    # 输出到console的log等级的开关
    ch.setLevel(logging.INFO)

    # 定义handler的输出格式
    # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s [%(thread)u] %(levelname)s: %(message)s")

    # 为文件输出设定格式
    fh.setFormatter(formatter)

    # 控制台输出设定格式
    ch.setFormatter(formatter)

    # 设置文件输出到logger
    logger.addHandler(fh)

    # 设置控制台输出到logger
    logger.addHandler(ch)
    return logger


logger = logger_init()