from __future__ import annotations

import logging
from logging import StreamHandler


class Logger:
    """
    Singleton logger class for hyperion
    """

    _instance = None

    def __new__(cls, *args: object, **kwargs: object) -> "Logger":
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.__logger = logging.Logger("hyperion")

        formatter: logging.Formatter = logging.Formatter("[%(asctime)s - %(name)s]: %(message)s", datefmt="%H:%M:%S")

        self.__logger.setLevel(logging.DEBUG)
        self.__logger.propagate = False
        handler: StreamHandler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)

    def debug(self, message: str, *args: object) -> None:
        """
        Log a debug message
        :param message: the message to log
        :return: nothing, will log a message
        """
        self.__logger.debug(message, *args)

    def info(self, message: str, *args: object) -> None:
        """
        Log an info message
        :param message: the message to log
        :return: nothing, will log a message
        """
        self.__logger.info(message, *args)

    def warning(self, message: str, *args: object) -> None:
        """
        Log a warning message
        :param message: the message to log
        :return: nothing, will log a message
        """
        self.__logger.warning(message, *args)

    def error(self, message: str, *args: object) -> None:
        """
        Log an error message
        :param message: the message to log
        :return: nothing, will log a message
        """
        self.__logger.error(message, *args)

    def critical(self, message: str, *args: object) -> None:
        """
        Log a critical message
        :param message: the message to log
        :return: nothing, will log a message
        """
        self.__logger.critical(message, *args)

    def exception(self, message: str, *args: object) -> None:
        """
        Log an exception message
        :param message: the message to log
        :return: nothing, will log a message
        """
        self.__logger.exception(message, *args)


logger = Logger()
