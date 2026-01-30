import os
import logging
import colorlog
import setproctitle
import sys
from transformers import TrainingArguments

def _extract_run_info(training_args: TrainingArguments | None):
    run_name = None
    model_name = None
    project_name = None

    if training_args is not None:
        # Run name from args, else try to derive from output_dir
        run_name = getattr(training_args, "run_name", None)
        if not run_name:
            out = getattr(training_args, "output_dir", None)
            if out:
                run_name = os.path.basename(os.path.abspath(out))

        # Model name preference: model_name_or_path -> model_name
        model_name = getattr(training_args, "model_name_or_path", None) or getattr(training_args, "model_name", None)

        # Project: env WANDB_PROJECT -> hf hub id -> run_name fallback
        project_name = os.environ.get("WANDB_PROJECT") or getattr(training_args, "hub_model_id", None) or run_name

    # Final fallbacks
    run_name = run_name or os.environ.get("RUN_NAME", "run")
    model_name = model_name or os.environ.get("MODEL_NAME", "unknown-model")
    project_name = project_name or os.environ.get("PROJECT", "default")

    return run_name, model_name, project_name


def set_process_title(training_args: TrainingArguments = None, *, model: str | None = None, project: str | None = None):
    """
    Sets the process title with format:
    AgentRL:{run_name} | py:{python} | model:{model} | proj:{project} | Rank {rank} ({script})
    """
    run_name, model_name_from_args, project_name_from_args = _extract_run_info(training_args)

    # Allow explicit override via parameters
    model_name = model or model_name_from_args
    project_name = project or project_name_from_args

    title_prefix = f"AgentRL:{run_name}"
    py_path = sys.executable
    rank = os.environ.get("RANK", "0")
    script_name = sys.argv[0]

    # Compose final title
    final_title = (
        f"{title_prefix} | py:{py_path} | model:{model_name} | proj:{project_name} | Rank {rank} ({script_name})"
    )
    setproctitle.setproctitle(final_title)

# Set a default title. It will be updated later in main.py
set_process_title()

logger = logging.getLogger("AgentRL")
logging_level = logging._nameToLevel[os.environ.get("LOG_LEVEL", "info").upper()]
logger.setLevel(logging_level)
# Prevent log propagation to root logger to avoid duplicate output
logger.propagate = False
stream_handler = colorlog.StreamHandler()
stream_handler.setLevel(logging_level)
formatter = colorlog.ColoredFormatter(
    '%(log_color)s'+'[RANK {}] '.format(os.environ.get("RANK", "0")) + '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
    log_colors={
        'DEBUG': 'bold_black',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

from transformers.trainer import logger as tf_logger
tf_logger.setLevel(logging_level)
tf_logger.propagate = False
tf_handler = colorlog.StreamHandler()
tf_handler.setLevel(logging_level if os.environ.get("RANK", "0") == "0" else logging.CRITICAL)
tf_formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(message)s',
    log_colors={
        'DEBUG': 'bold_black',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    })
tf_handler.setFormatter(tf_formatter)
tf_logger.addHandler(tf_handler)