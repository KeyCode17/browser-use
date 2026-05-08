"""
End-to-end PoC: browser-use Agent driven by

  1. Claude Code OAuth credentials  (no ANTHROPIC_API_KEY, billed against your
     Claude subscription via the local `claude` CLI's stored token), and
  2. an externally-launched Chromium-family browser over CDP (cosmium drop-in
     point — for now any Chromium will do).

Run the browser separately so its lifetime is decoupled from the agent:

    chromium --remote-debugging-port=9222 \
             --user-data-dir=/tmp/cdp-profile \
             --no-first-run

Then verify CDP is up:

    curl -s http://127.0.0.1:9222/json/version | jq .

Then run this script:

    uv run python examples/claude_code_oauth_cosmium.py
"""

import asyncio
import os

from browser_use import Agent, BrowserSession
from browser_use.llm.anthropic.claude_code import ChatAnthropicClaudeCode

CDP_URL = os.environ.get('CDP_URL', 'http://127.0.0.1:9222')
TASK = os.environ.get('TASK', 'Open https://example.com and report the page heading.')


async def main() -> None:
	llm = ChatAnthropicClaudeCode(model='claude-sonnet-4-5')
	print(f'[ok] Claude Code OAuth resolved, cc_version={llm.cc_version}')

	session = BrowserSession(cdp_url=CDP_URL)
	print(f'[ok] BrowserSession will attach to {CDP_URL}')

	agent = Agent(task=TASK, llm=llm, browser_session=session)
	history = await agent.run()
	print('[done]', history.final_result())


if __name__ == '__main__':
	asyncio.run(main())
