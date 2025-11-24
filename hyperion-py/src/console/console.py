from __future__ import annotations
from enum import Enum
from typing import Any


class ANSI(Enum):
    # Reset
    RESET = "\033[0m"

    # Text Colours
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Styling
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    REVERSED = "\033[7m"


class ConsoleFormatter:
    class _Format:
        def __init__(self, format_str: str = "", is_exit: bool = False) -> None:
            self.format: str = format_str
            self.is_exit: bool = is_exit

        def add_format(self, format_str: str) -> None:
            self.format: str = self.format + "" + format_str

        def create_clear(self) -> ConsoleFormatter._Format:
            return ConsoleFormatter._Format(self.format, is_exit=True)

        def __str__(self) -> str:
            return ANSI.RESET.value if self.is_exit else self.format.strip()

    def __init__(self) -> None:
        self.__instructions: list[str | ConsoleFormatter._Format] = []
        self.__current_formatting: ConsoleFormatter._Format | None = None

    def __str__(self) -> str:
         return self.build()

    @staticmethod
    def warning(message: str, indentation: int = 0, new_lines_before_message: int = 0) -> None:
        ConsoleFormatter().add("\n" * new_lines_before_message).add("\t" * indentation).add_warning_mark().apply_bold_yellow(message).build_and_print()

    @staticmethod
    def error(message: str, indentation: int = 0, new_lines_before_message: int = 0) -> None:
        ConsoleFormatter().add("\n" * new_lines_before_message).add("\t" * indentation).add_exclamation_mark().apply_bold_red(message).build_and_print()

    @staticmethod
    def success(message: str, indentation: int = 0, new_lines_before_message: int = 0) -> None:
        ConsoleFormatter().add("\n" * new_lines_before_message).add("\t" * indentation).add_check_mark().apply_bold_green(message).build_and_print()

    @staticmethod
    def info(message: str, indentation: int = 0, new_lines_before_message: int = 0) -> None:
        ConsoleFormatter().add("\n" * new_lines_before_message).add("\t" * indentation).add(message).build_and_print()

    @staticmethod
    def new_section(message: str, new_lines_before_message: int = 0) -> None:
        (ConsoleFormatter()
         .add('\n' * new_lines_before_message)
         .apply_bold_magenta("=" * 60)
         .add_newline()
         .apply_bold_magenta(message)
         .add_newline()
         .apply_bold_magenta("=" * 60)
         .build_and_print())

    def set_formatting(self, format_str: str) -> ConsoleFormatter:
        self.__current_formatting = ConsoleFormatter._Format(format_str)
        self.__instructions.append(format_str)
        return self

    def add_formatting(self, format_str: str) -> ConsoleFormatter:
        if self.__current_formatting is None:
            self.__current_formatting = ConsoleFormatter._Format()
            self.__instructions.append(self.__current_formatting)
        self.__current_formatting.add_format(format_str)
        return self

    def add_emoji(self, emoji: str) -> ConsoleFormatter:
        return self.add(f" {emoji} ")

    def add_satellite(self) -> ConsoleFormatter:
        return self.add_emoji("🛰️")

    def add_check_mark(self) -> ConsoleFormatter:
        return self.add_emoji("️✅")

    def add_exclamation_mark(self) -> ConsoleFormatter:
        return self.add_emoji("❗️")

    def add_warning_mark(self) -> ConsoleFormatter:
        return self.add_emoji("⚠️")

    def add_tab(self) -> ConsoleFormatter:
        return self.add("\t")

    def add_newline(self) -> ConsoleFormatter:
        return self.add("\n")

    def apply_bold(self) -> ConsoleFormatter:
        return self.add_formatting(ANSI.BOLD.value)

    def apply_cyan(self) -> ConsoleFormatter:
        return self.add_formatting(ANSI.CYAN.value)

    def apply_magenta(self) -> ConsoleFormatter:
        return self.add_formatting(ANSI.MAGENTA.value)

    def apply_green(self) -> ConsoleFormatter:
        return self.add_formatting(ANSI.GREEN.value)

    def apply_red(self) -> ConsoleFormatter:
        return self.add_formatting(ANSI.RED.value)

    def apply_yellow(self) -> ConsoleFormatter:
        return self.add_formatting(ANSI.YELLOW.value)

    def apply_bold_cyan(self, message: Any | None = None) -> ConsoleFormatter:
        if message is None:
            return self.apply_bold().apply_cyan()
        return self.apply_bold().apply_cyan().add(message).clear_formatting()

    def apply_bold_magenta(self, message: Any | None = None) -> ConsoleFormatter:
        if message is None:
            return self.apply_bold().apply_magenta()
        return self.apply_bold().apply_magenta().add(message).clear_formatting()

    def apply_bold_red(self, message: Any | None = None) -> ConsoleFormatter:
        if message is None:
            return self.apply_bold().apply_red()
        return self.apply_bold().apply_red().add(message).clear_formatting()

    def apply_bold_green(self, message: Any | None = None) -> ConsoleFormatter:
        if message is None:
            return self.apply_bold().apply_green()
        return self.apply_bold().apply_green().add(message).clear_formatting()

    def apply_bold_yellow(self, message: Any | None = None) -> ConsoleFormatter:
        if message is None:
            return self.apply_bold().apply_yellow()
        return self.apply_bold().apply_yellow().add(message).clear_formatting()

    def clear_formatting(self) -> ConsoleFormatter:
        if self.__instructions is None:
            return self
        self.__instructions.append(self.__current_formatting.create_clear())
        self.__current_formatting = None
        return self

    def add(self, message: Any) -> ConsoleFormatter:
        self.__instructions.append(str(message))
        return self

    def build(self) -> str:
        if self.__current_formatting is not None:
            self.clear_formatting()
        return "".join(str(i) for i in self.__instructions)

    def build_and_print(self) -> None:
        print(self.build())
