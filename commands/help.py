from importlib import import_module

description = (
    'List commands, or show details for one. ~admin lists the gated commands.\n'
    '  ~help [page|command]'
)

# Commands that work but are intentionally NOT listed by ~help (still reachable
# via ~help <command>).
HIDDEN = {"markov", "mimic", "admin"}

# Admin / super-admin commands — listed only by ~admin (or ~help admin), never
# in the public ~help.
ADMIN_CATEGORIES = [
    ("super admin", [
        "botpersona", "botmode", "botcontext", "botchance",
        "banuser", "unbanuser", "warnings", "modelqueue",
        "chan_autotl", "global_autotl", "notranslate",
    ]),
    ("admin", [
        "autotl", "setlang", "tloutput",
    ]),
]
ADMIN_NAMES = {name for _title, names in ADMIN_CATEGORIES for name in names}

# Public categories shown by ~help (admin + hidden commands excluded).
CATEGORIES = [
    ("translation", [
        "practice", "romanize", "speak",
    ]),
    ("archive", [
        "said", "saidnext", "saidmost", "regex", "userregex", "quote", "random",
        "firstseen", "chatstats", "regulars", "askchat",
    ]),
    ("personas", [
        "persona", "hyper", "generate",
    ]),
    ("analysis", [
        "whosaid", "markers", "like", "vibes", "twin", "traits", "style",
        "top", "bottom", "distinct", "why", "axis", "emote", "explain",
        "iq", "funny", "irony",
    ]),
    ("utility", [
        "help", "experimental", "ping", "artifacts",
    ]),
]

GROUPS_PER_PAGE = 2

# Rough/prototype commands get an ε marker in the listing (legend appended
# when any are shown; full notes via ~experimental). Top-level names only —
# subroutes like "explain emote" are covered by their parent.
EXPERIMENTAL_MARK = {"askchat", "emote", "irony", "iq"}


def _known_command_pages(command_names):
    grouped = []
    seen = set()
    # hidden + admin commands never appear in the public listing
    command_set = set(command_names) - HIDDEN - ADMIN_NAMES
    for title, names in CATEGORIES:
        present = [name for name in names if name in command_set]
        if present:
            grouped.append((title, present))
            seen.update(present)
    leftovers = sorted(command_set - seen)
    if leftovers:
        grouped.append(("other", leftovers))
    return [
        grouped[index:index + GROUPS_PER_PAGE]
        for index in range(0, len(grouped), GROUPS_PER_PAGE)
    ]


def _format_command_list(prefix: str, page: int, pages) -> str:
    total = max(1, len(pages))
    page = max(1, min(page, total))
    chunks = []
    marked = False
    for title, names in pages[page - 1]:
        rendered = " ".join(
            f"{prefix}{name}{'ε' if name in EXPERIMENTAL_MARK else ''}"
            for name in names)
        marked = marked or any(name in EXPERIMENTAL_MARK for name in names)
        chunks.append(f"{title}: {rendered}")
    legend = f" | ε=experimental ({prefix}experimental)" if marked else ""
    return (
        f"Commands {page}/{total}: " + " | ".join(chunks) +
        f" | {prefix}help <command> for details{legend}"
    )


def format_admin_list(prefix: str, command_names) -> str:
    command_set = set(command_names)
    chunks = []
    for title, names in ADMIN_CATEGORIES:
        present = [n for n in names if n in command_set]
        if present:
            rendered = " ".join(f"{prefix}{n}" for n in present)
            chunks.append(f"{title}: {rendered}")
    if not chunks:
        return "No admin commands available."
    return (
        "Admin commands — " + " | ".join(chunks) +
        f" | {prefix}help <command> for details"
    )


async def handle_help(bot, message, params):
    from command_registry import command_handlers

    if not params:
        pages = _known_command_pages(command_handlers.keys())
        await message.channel.send(_format_command_list(bot.prefix, 1, pages))
        return

    command = params[0].lower()
    if command == "admin":
        await message.channel.send(format_admin_list(bot.prefix, command_handlers.keys()))
        return
    if command in {"experimental", "experiments"}:
        module = import_module("commands.experimental")
        await message.channel.send(module.format_experimental_list(bot.prefix))
        return
    if command.isdigit():
        pages = _known_command_pages(command_handlers.keys())
        await message.channel.send(_format_command_list(bot.prefix, int(command), pages))
        return
    if command in command_handlers:
        try:
            module = import_module(f"commands.{command}")
            description = getattr(module, 'description', "No description available for this command.")
        except ImportError as e:
            description = f"Failed to load command module: {str(e)}"
        await message.channel.send(f"{command}: {description}")
    else:
        await message.channel.send(f"Command not found. Use {bot.prefix}help to list all commands.")
