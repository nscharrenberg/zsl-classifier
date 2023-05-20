import colorama
import emoji
from colorama import Fore

ERROR = Fore.RED
INFO = Fore.CYAN
WARNING = Fore.YELLOW
OK = Fore.GREEN
DEFAULT = Fore.RESET


def log(message: str, level: int = 0, verbose: bool = True):
    """
    When allowed, print a log with the given message and log level.

    :param message: The message to log
    :param level: The level of the log (DEBUG = 0, OK = 10, INFO = 20, WARN = 30, ERROR = 40, CRITICAL = 50)
    :param verbose: Whether to log.
    """
    if not verbose:
        return

    color = ""
    icon = ":white_circle:"
    if level == 10:
        color = OK
        icon = ":check_mark:"
    elif level == 20:
        color = INFO
    elif level == 30:
        color = WARNING
    elif level >= 40:
        color = ERROR
        icon = ":cross_mark:"

    print(emoji.emojize(f"{color}{icon} {message}{DEFAULT}", variant="emoji_type", language='alias'))
