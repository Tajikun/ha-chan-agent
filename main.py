from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from pydantic import BaseModel
import discord
from discord import app_commands
import os
import io
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Tuple


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


@tool
def search_tool(query: str) -> str:
    """A tool to search for contact information in a database."""
    return f"Results for: {query}"


agent = create_agent(
    model = "gpt-4o",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)


TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()


# Helpers
def _parse_category_spec(guild: discord.Guild, spec: str) -> List[discord.CategoryChannel]:
    """
    Accepts a comma-separated list of category names or IDs.
    Returns matching CategoryChannel objects (deduped, preserving order).
    """
    if not spec:
        return []

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    seen: set[int] = set()
    out: List[discord.CategoryChannel] = []

    # First pass: IDs
    for p in parts:
        if p.isdigit():
            ch = guild.get_channel(int(p))
            if isinstance(ch, discord.CategoryChannel) and ch.id not in seen:
                seen.add(ch.id)
                out.append(ch)

    # Second pass: names (case-insensitive exact match)
    for p in parts:
        if p.isdigit():
            continue
        for cat in guild.categories:
            if cat.name.lower() == p.lower() and cat.id not in seen:
                seen.add(cat.id)
                out.append(cat)

    return out

async def _search_channel_for_mentions(
    channel: discord.TextChannel,
    user: discord.User | discord.Member,
    after_dt: datetime | None,
    per_channel_limit: int,
) -> List[discord.Message]:
    """Collect messages in `channel` where `user` was mentioned."""
    hits: List[discord.Message] = []
    kwargs: dict = {"limit": per_channel_limit}
    if after_dt:
        kwargs["after"] = after_dt

    try:
        async for msg in channel.history(**kwargs):
            if user in msg.mentions:
                hits.append(msg)
            elif f"<@{user.id}>" in msg.content or f"<@!{user.id}>" in msg.content:
                hits.append(msg)
    except discord.Forbidden:
        pass
    except discord.HTTPException:
        pass

    return hits


async def _search_active_threads_for_mentions(
    channel: discord.TextChannel,
    user: discord.User | discord.Member,
    after_dt: datetime | None,
    per_thread_limit: int,
) -> List[Tuple[discord.Thread, List[discord.Message]]]:
    """Search only active (unarchived) threads under a text channel."""
    results: List[Tuple[discord.Thread, List[discord.Message]]] = []
    for th in channel.threads:
        if th.locked:
            continue
        try:
            th_hits: List[discord.Message] = []
            kwargs: dict = {"limit": per_thread_limit}
            if after_dt:
                kwargs["after"] = after_dt
            async for msg in th.history(**kwargs):
                if user in msg.mentions or f"<@{user.id}>" in msg.content or f"<@!{user.id}>" in msg.content:
                    th_hits.append(msg)
            if th_hits:
                results.append((th, th_hits))
        except (discord.Forbidden, discord.HTTPException):
            continue
    return results
    

EMBED_FIELD_LIMIT = 1024
EMBED_MAX_FIELDS = 25  # absolute Discord limit

def chunk_lines_into_fields(lines, field_name_prefix="Summary"):
    """
    Pack lines into <=1024-char field values.
    Returns a list of (name, value) pairs ready for embed.add_field.
    """
    fields = []
    buf = []
    buf_len = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # +1 for the newline that will be inserted on join
        add_len = len(line) + (1 if buf else 0)
        if buf_len + add_len > EMBED_FIELD_LIMIT:
            fields.append((f"{field_name_prefix} {len(fields)+1}", "\n".join(buf)))
            buf = [line]
            buf_len = len(line)
        else:
            buf.append(line)
            buf_len += add_len
    if buf:
        fields.append((f"{field_name_prefix} {len(fields)+1}", "\n".join(buf)))
    return fields


class AgentClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

client = AgentClient()


@client.tree.command(name="find-collabs", description="Find collabs user is mentioned in.")
@app_commands.describe(
    categories="Comma-separated list of category names or IDs to limit search to.",
    days="Number of days in the past to search (default 7).",
    per_channel_limit="Max messages to search per channel (default 200).",
    include_threads="Whether to include active threads in the search (default False).",
)
async def find_collabs(
    interaction: discord.Interaction,
    categories: str = "",
    days: int = 7,
    per_channel_limit: int = 200,
    include_threads: bool = False,
):
    await interaction.response.defer(thinking=True, ephemeral=True)

    guild = interaction.guild
    if guild is None:
        await interaction.followup.send("This command must be used in a server.", ephemeral=True)
        return

    # Resolve categories
    if categories.strip():
        target_categories = _parse_category_spec(guild, categories)
        if not target_categories:
            await interaction.followup.send(
                "I couldn't match any categories from your input. "
                "Provide comma-separated category names or IDs.",
                ephemeral=True,
            )
            return
    else:
        target_categories = list(guild.categories)

    # Time window
    after_dt = datetime.now(timezone.utc) - timedelta(days=max(0, days))

    # Collect results
    invoker = interaction.user
    total_hits = 0
    lines: List[str] = []
    MAX_LINES = 40  # to keep the embed readable; overflow goes to a file

    for cat in target_categories:
        cat_header_added = False

        # Search text channels in this category
        for ch in cat.text_channels:
            hits = await _search_channel_for_mentions(ch, invoker, after_dt, per_channel_limit)
            if hits:
                total_hits += len(hits)
                if not cat_header_added:
                    lines.append(f"**Category:** {cat.name} (`{cat.id}`)")
                    cat_header_added = True
                # Summarize per-channel with first few links
                # Discord message jump URLs are perfect for navigation
                sample = hits[:3]
                sample_links = ", ".join(f"[link]({m.jump_url})" for m in sample)
                lines.append(f"• <#{ch.id}> — {len(hits)} mention(s): {sample_links}{' …' if len(hits) > 3 else ''}")

                # Optional: Threads
                if include_threads:
                    thread_hits = await _search_active_threads_for_mentions(ch, invoker, after_dt, min(per_channel_limit, 100))
                    for th, th_msgs in thread_hits:
                        total_hits += len(th_msgs)
                        sample_th = th_msgs[:3]
                        sample_th_links = ", ".join(f"[link]({m.jump_url})" for m in sample_th)
                        lines.append(
                            f"   ↳ 🧵 <#{th.id}> — {len(th_msgs)} mention(s): {sample_th_links}{' …' if len(th_msgs) > 3 else ''}"
                        )

                if len(lines) >= MAX_LINES:
                    lines.append("_Output truncated; see attachment for the full list._")
                    break
        if len(lines) >= MAX_LINES:
            break

    if total_hits == 0:
        await interaction.followup.send("No pings found for you in the selected categories and time window.", ephemeral=True)
        return

    embed = discord.Embed(
        title="Mentions Found",
        description=f"Found **{total_hits}** mention(s) for {invoker.mention} in the last **{days}** day(s).",
    )

    fields = chunk_lines_into_fields(lines, field_name_prefix="Results")
    file_obj = None

    if len(fields) >= EMBED_MAX_FIELDS:
        # Too many fields; attach full text and keep the first few fields
        full_text = "\n".join(lines)
        file_obj = discord.File(io.BytesIO(full_text.encode("utf-8")), filename="mentions_report.txt")

        # Leave room for other fields if you add more later; keep e.g. first 10
        kept = min(10, EMBED_MAX_FIELDS - 1)
        for name, value in fields[:kept]:
            embed.add_field(name=name, value=value, inline=False)
        embed.add_field(
            name="Note",
            value="Output truncated in embed; see attached `mentions_report.txt` for the full list.",
            inline=False,
        )
    else:
        for name, value in fields:
            embed.add_field(name=name, value=value, inline=False)

    # --- Send (do NOT pass file=None)
    if file_obj:
        await interaction.followup.send(embed=embed, file=file_obj, ephemeral=True)
    else:
        await interaction.followup.send(embed=embed, ephemeral=True)


@client.tree.command(name="ha-chat", description="Ask ha-chan a question.")
@app_commands.describe(message="What would you like to ask ha-chan?")
async def ask(interaction: discord.Interaction, message: str):
    await interaction.response.defer(thinking=True)

    def run_agent_call():
        result = agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })
        structured = result.get("structured_response")
        try:
            if isinstance(structured, BaseModel):
                return structured.model_dump()
            if isinstance(structured, dict):
                return structured
            return str(structured)
            
        except Exception:
            return str(structured)
        

    output = await asyncio.to_thread(run_agent_call)

    embed = discord.Embed(title="Ha-chan's Response")
    if isinstance(output, dict):
        for k, v in output.items():
            embed.add_field(name=k.capitalize(), value=str(v), inline=False)
    else:
        embed.description = str(output)

    await interaction.followup.send(embed=embed)


# Running
if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable not set.")
    client.run(TOKEN)