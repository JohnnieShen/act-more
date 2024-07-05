'''
日志记录配置

      "format": "%(asctime)s - %(module)s - %(name)s - %(process)d - %(thread)d - %(levelname)s - %(message)s"

'''
import sys

sys.path.append('/home/wedo/opt/fr5eg')

import json
import logging.config
import os
# import NFCWebots.constants as const


class logging_main:
    @staticmethod
    def setup_logging(default_path="config/inner_logging.json", default_level=logging.INFO):
        '''
        加载logging对应的JSON配置文件
        :param default_path: logging.json的配置文件路径和名称
        :param default_level: 设定默认的记录日志的级别
        :return:
        '''
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 定义日志文件夹和文件名
        # log_folder = os.path.join(script_dir, 'log')
        # # 检查日志文件夹是否存在，如果不存在则创建
        # if not os.path.exists(log_folder):
        #     os.makedirs(log_folder)

        dirFileName = os.path.join(script_dir, default_path)
        path = dirFileName

        if os.path.exists(path):
            with open(path, const.FILE_ONLY_READ, encoding=const.ECODING_UTF8) as f:
                config = json.load(f)
                # 通过JSON加载配置文件，然后通过logging.dictConfig配置logging
                logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
        return logging

    def func(self):
        logging.info("start func")
        logging.info("exec func")
        logging.info("end func")


# # 假设你的JSON配置文件路径为'logging_config.json'
# logging_main.setup_logging(default_path="inner_logging.json")

if __name__ == "__main__":
    logging = logging_main.setup_logging(default_path="config/inner_logging.json")
    # 获取logger实例
    logger = logging.getLogger("main_log")
    logging_main().func()
