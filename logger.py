import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to terminal output based on log level."""
    
    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    
    FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: GREEN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET
    }

    def format(self, record):
        logfmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(logfmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def getlogger():
    """
    Sets up a dual-output ROOT logger:
    - Prints COLORED INFO, WARNING, and ERROR to the terminal.
    - Saves PLAIN TEXT DEBUG, INFO, WARNING, and ERROR to an absolute log file.
    """
    # FIX 1: By passing None, we configure the ROOT logger. 
    # Now, any file that calls logging.getLogger(__name__) will automatically 
    # inherit these console and file handlers.
    logger = logging.getLogger()
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG) 

        # Console Handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO) 
        console.setFormatter(ColoredFormatter()) 

        plain_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # FIX 2: Anchor the log directory using an absolute path based on where THIS script lives.
        # This completely nullifies working-directory path ghosts on the server.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logdir = os.path.join(base_dir, "DSpinGNNLogs")
        os.makedirs(logdir, exist_ok=True)

        utcnow = datetime.now(ZoneInfo("UTC"))
        localdt = utcnow.astimezone(ZoneInfo("Asia/Karachi"))
        logfile = f"DSpinGNN_{localdt.strftime('%Y%m%d_%H%M%S')}.log"
        
        # FIX 3: flush the file immediately upon write. 
        # If the server crashes via OOM or PyG C++ error, the logs are safely saved.
        file = logging.FileHandler(os.path.join(logdir, logfile), delay=False)
        file.setLevel(logging.DEBUG) 
        file.setFormatter(plain_formatter) 

        logger.addHandler(console)
        logger.addHandler(file)

    return logger