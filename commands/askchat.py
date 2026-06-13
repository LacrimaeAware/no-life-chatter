import asyncio

from services import llm
from utils import archive_qa
from utils.output_filter import is_clean

description = (
    "Evidence-backed archive/lore question answering over claims, chat lines, and emotes.\n"
    "  ~askchat [raw] [user=<name>|<name>] [chat=<channel>|chat=here] <question>"
)


async def handle_askchat(bot, message, params):
    raw = False
    if params and params[0].lower() in {"raw", "evidence", "receipts", "noai"}:
        raw = True
        params = params[1:]
    parsed = archive_qa.parse_params(params or [], getattr(message.channel, "name", None))
    if not parsed["query"]:
        await message.channel.send(
            "Usage: ~askchat [raw] [user=<name>|<name>] [chat=<channel>|chat=here] <question>"
        )
        return
    report = await asyncio.to_thread(
        archive_qa.build_report,
        parsed["query"],
        author=parsed["author"],
        channel=parsed["channel"],
    )
    fallback = archive_qa.format_chat(report)
    if raw or not archive_qa.has_strong_evidence(report):
        await message.channel.send(fallback)
        return

    if await llm.available():
        answer = await llm.chat(
            archive_qa.answer_messages(report),
            max_tokens=110,
            temperature=0.2,
        )
        formatted = archive_qa.format_answer_chat(report, answer or "")
        if formatted and is_clean(formatted):
            await message.channel.send(formatted)
            return

    await message.channel.send(fallback)
