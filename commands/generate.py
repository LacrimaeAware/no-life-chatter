from utils import persona_generate
from utils.output_filter import is_clean

description = (
    "Generate an example message from any mix of tags: chatter names (several "
    "= a fusion), trait poles (optimist/doomer/professor/...), chat=<channel>, "
    "year=<YYYY>, topic words, engine=markov|llm, model=llama|lora. Save "
    "per-user recipes and stack tags on top of them.\n"
    "  ~generate <tags...>            e.g. ~generate someuser doomer chat=somechannel\n"
    "  ~generate save <name> <tags...> · ~generate <name> [more tags] · "
    "~generate list · ~generate del <name>"
)


async def handle_generate(bot, message, params):
    if not params:
        await message.channel.send(
            "Usage: ~generate <tags...> (chatters, traits, chat=, year=, topic) "
            "| save <name> <tags> | list | del <name>")
        return
    user = message.author.name if message.author else "unknown"
    sub = params[0].lower()

    if sub == "save":
        if len(params) < 3:
            await message.channel.send("Usage: ~generate save <name> <tags...>")
            return
        err = persona_generate.save_combo(user, params[1], params[2:])
        await message.channel.send(err or f"saved '{params[1].lower()}' for you 💾")
        return
    if sub == "list":
        combos = persona_generate.list_combos(user)
        if not combos:
            await message.channel.send("You have no saved recipes. ~generate save <name> <tags...>")
            return
        await message.channel.send(
            "Your recipes: " + " · ".join(f"{n} ({t})" for n, t in combos[:8]))
        return
    if sub in ("del", "delete"):
        if len(params) < 2:
            await message.channel.send("Usage: ~generate del <name>")
            return
        ok = persona_generate.delete_combo(user, params[1])
        await message.channel.send("deleted 🗑️" if ok else "no recipe by that name.")
        return

    recipe = persona_generate.parse_recipe(params, user)
    out, err = await persona_generate.generate_example(recipe)
    if not out:
        await message.channel.send(f"Couldn't generate: {err}")
        return
    if not is_clean(out):
        await message.channel.send("(generated something I won't repeat — try again)")
        return
    label_bits = recipe["users"] + recipe["traits"]
    if recipe["expanded"]:
        label_bits = recipe["expanded"] + [t for t in label_bits if t not in recipe["users"]] \
            if recipe["users"] else recipe["expanded"] + recipe["traits"]
    label = "+".join(label_bits) if label_bits else (recipe["topic"][:24] or "example")
    if len(out) > 440:
        out = out[:439] + "..."
    await message.channel.send(f"⚗️ {label}: {out}")
    from utils import reaction_tracker
    reaction_tracker.watch(message.channel.name, out,
                           {"kind": "generate", "recipe": label})
