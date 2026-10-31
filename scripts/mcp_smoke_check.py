"""Drives embedding_service.mcp_server as a real subprocess over stdio -
the actual deployed path an MCP client uses - to verify search_corpus
really pulls docs/corpus.txt, encodes it with the real on-device model,
and ranks it against a real prompt. Not a pytest test (spawns a subprocess
that loads the full model - slow); run manually: python3 scripts/mcp_smoke_check.py
"""

import asyncio
import os

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command="python3",
        args=["-m", "embedding_service.mcp_server"],
        env={**os.environ, "PYTHONPATH": "src"},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print("Registered tools:", names)
            assert names == ["corpus_info", "embed_text", "reload_corpus", "search_corpus"], names

            info = await session.call_tool("corpus_info", {})
            print("corpus_info ->", info.content[0].text)

            query = "What causes the northern lights?"
            result = await session.call_tool("search_corpus", {"query": query, "top_k": 3})
            print(f"search_corpus({query!r}) ->")
            for block in result.content:
                print(" ", block.text)

    print("\nOK - MCP server actually pulled the corpus and ranked it against a real prompt with the real model.")


if __name__ == "__main__":
    asyncio.run(main())
