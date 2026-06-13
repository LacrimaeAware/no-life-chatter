import importlib
import logging
import os
import traceback

command_handlers = {}
command_load_errors = {}

# Resolve the commands directory relative to this file so the bot works no
# matter what the current working directory is.
COMMAND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commands")

def load_command_handlers():
    global command_handlers, command_load_errors
    command_handlers.clear()  # Clear existing handlers
    command_load_errors.clear()
    command_files = [f[:-3] for f in os.listdir(COMMAND_DIR) if f.endswith('.py') and not f.startswith('__')]

    for cmd_file in command_files:
        try:
            module = importlib.import_module(f"commands.{cmd_file}")
        except Exception as exc:
            logging.exception(f"Failed to load command module {cmd_file}")
            command_load_errors[cmd_file] = {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            continue
        handler_function_name = f'handle_{cmd_file}'
        handler = getattr(module, handler_function_name, None)

        if handler:
            command_handlers[cmd_file] = handler
        else:
            logging.warning(f"Handler function {handler_function_name} not found in {cmd_file}")
            command_load_errors[cmd_file] = {
                "error": f"missing {handler_function_name}",
                "traceback": "",
            }

# Initial load
load_command_handlers()
