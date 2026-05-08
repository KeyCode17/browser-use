"""
ChatAnthropicClaudeCode — reuse the local Claude Code CLI's OAuth login as the
LLM credential, so browser-use agents bill against the user's Claude
subscription quota instead of API credits.

Auth resolution order (matches the claude-rust port):
    1. ANTHROPIC_API_KEY env  (caller wants raw API; bail out, use ChatAnthropic instead)
    2. macOS Keychain         (`security find-generic-password -s "Claude Code-credentials" -w`)
    3. ~/.claude/.credentials.json   <- the Linux path
    4. ~/.claude/settings.json       (env.ANTHROPIC_AUTH_TOKEN)

The OAuth path requires a coordinated set of headers + a body-level "billing
header" system block that tells Anthropic's backend to route this call to
subscription billing instead of pay-per-token. Drop any one piece -> 401/403
or silent rebill against API credits.

Version drift: the User-Agent and `cc_version` track the installed `claude`
CLI version (resolved from `claude --version` at startup), so updating the CLI
keeps headers fresh without library changes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from browser_use.llm.anthropic.chat import ChatAnthropic
from browser_use.llm.messages import BaseMessage, ContentPartTextParam, SystemMessage

ANTHROPIC_BASE_URL = 'https://api.anthropic.com'
ANTHROPIC_VERSION = '2023-06-01'
OAUTH_BETA = 'oauth-2025-04-20,interleaved-thinking-2025-05-14,claude-code-20250219,prompt-caching-2024-07-31'
FALLBACK_CC_VERSION = '2.1.133'  # bump if `claude` not installed at runtime


class ClaudeCodeAuthError(RuntimeError):
	pass


def _read_credentials_file() -> dict | None:
	path = Path.home() / '.claude' / '.credentials.json'
	if not path.exists():
		return None
	try:
		return json.loads(path.read_text()).get('claudeAiOauth')
	except (OSError, json.JSONDecodeError):
		return None


def _read_macos_keychain() -> dict | None:
	if sys.platform != 'darwin':
		return None
	try:
		out = subprocess.run(
			['security', 'find-generic-password', '-s', 'Claude Code-credentials', '-w'],
			capture_output=True,
			text=True,
			timeout=5,
		)
	except (OSError, subprocess.TimeoutExpired):
		return None
	if out.returncode != 0:
		return None
	try:
		return json.loads(out.stdout.strip()).get('claudeAiOauth')
	except json.JSONDecodeError:
		return None


def _read_settings_token() -> tuple[str, str] | None:
	"""Returns (auth_token, base_url) from ~/.claude/settings.json env block, or None."""
	path = Path.home() / '.claude' / 'settings.json'
	if not path.exists():
		return None
	try:
		env = json.loads(path.read_text()).get('env', {})
	except (OSError, json.JSONDecodeError):
		return None
	token = env.get('ANTHROPIC_AUTH_TOKEN')
	if not token:
		return None
	base = env.get('ANTHROPIC_BASE_URL') or ANTHROPIC_BASE_URL
	return token, base.rstrip('/')


def _resolve_oauth_token() -> tuple[str, int]:
	"""Returns (access_token, expires_at_ms). Raises ClaudeCodeAuthError on failure."""
	for source_name, reader in (('keychain', _read_macos_keychain), ('credentials.json', _read_credentials_file)):
		oauth = reader()
		if not oauth:
			continue
		access = oauth.get('accessToken')
		expires_at = int(oauth.get('expiresAt') or 0)
		if not access:
			continue
		if expires_at and time.time() * 1000 > expires_at:
			raise ClaudeCodeAuthError(
				f'Claude Code OAuth token from {source_name} is expired. Run `claude` to refresh, then retry.'
			)
		return access, expires_at
	raise ClaudeCodeAuthError(
		'No Claude Code OAuth credentials found. Run `claude` to log in, or set ANTHROPIC_API_KEY for the regular API path.'
	)


_VERSION_RE = re.compile(r'(\d+\.\d+\.\d+)')


def _resolve_cc_version() -> str:
	"""Reads `claude --version` and extracts the semver. Falls back to a pinned constant."""
	try:
		out = subprocess.run(['claude', '--version'], capture_output=True, text=True, timeout=5)
	except (OSError, subprocess.TimeoutExpired):
		return FALLBACK_CC_VERSION
	if out.returncode != 0:
		return FALLBACK_CC_VERSION
	match = _VERSION_RE.search(out.stdout)
	return match.group(1) if match else FALLBACK_CC_VERSION


def _build_default_headers(cc_version: str) -> dict[str, str]:
	return {
		'anthropic-version': ANTHROPIC_VERSION,
		'anthropic-beta': OAUTH_BETA,
		'anthropic-dangerous-direct-browser-access': 'true',
		'User-Agent': f'claude-cli/{cc_version} (external, cli)',
		'x-app': 'cli',
	}


def _billing_header_text(cc_version: str) -> str:
	# Matches the BILLING_HEADER_LINE constant from claude-rust-provider/anthropic_provider.rs.
	# The `.dNN` build suffix is omitted -- backend appears to accept bare semver, and we don't
	# have a stable way to read the suffix from the installed CLI.
	return f'x-anthropic-billing-header: cc_version={cc_version}; cc_entrypoint=cli;'


@dataclass
class ChatAnthropicClaudeCode(ChatAnthropic):
	"""
	ChatAnthropic variant that authenticates with the user's Claude Code OAuth
	session instead of an API key.

	Usage:
	    llm = ChatAnthropicClaudeCode(model='claude-sonnet-4-5')
	    agent = Agent(task='...', llm=llm)

	Default model is the OAuth-friendly Sonnet alias used by the CLI. Override
	via the `model` constructor arg.
	"""

	model: str = 'claude-sonnet-4-5'
	cc_version: str | None = None
	_billing_text: str = field(default='', init=False, repr=False)

	def __post_init__(self) -> None:
		# Skip if the caller pre-populated auth_token (test injection / custom flow).
		if not self.auth_token:
			if os.environ.get('ANTHROPIC_API_KEY'):
				raise ClaudeCodeAuthError(
					'ANTHROPIC_API_KEY is set; use ChatAnthropic for the regular API path. '
					'Unset it to use Claude Code OAuth.'
				)
			# settings.json is the third-tier "I already have a token" path.
			from_settings = _read_settings_token()
			if from_settings:
				token, base = from_settings
				self.auth_token = token
				self.base_url = self.base_url or base
			else:
				token, _expires = _resolve_oauth_token()
				self.auth_token = token

		if not self.base_url:
			self.base_url = ANTHROPIC_BASE_URL
		# Force api_key off so the SDK uses Bearer-only auth.
		self.api_key = None

		resolved_version = self.cc_version or _resolve_cc_version()
		self.cc_version = resolved_version
		self._billing_text = _billing_header_text(resolved_version)

		# Merge into any caller-supplied headers, but our values win.
		merged = dict(self.default_headers or {})
		merged.update(_build_default_headers(resolved_version))
		self.default_headers = merged

	@property
	def provider(self) -> str:
		return 'anthropic_claude_code'

	def _inject_billing_header(self, messages: list[BaseMessage]) -> list[BaseMessage]:
		"""Prepend the billing-header line to the system message so backend bills the
		Claude subscription instead of API credits. If no system message exists, insert one."""
		billing = self._billing_text
		new_messages = list(messages)
		first = new_messages[0] if new_messages else None
		if isinstance(first, SystemMessage):
			if isinstance(first.content, str):
				new_messages[0] = SystemMessage(content=f'{billing}\n\n{first.content}', cache=first.cache)
			else:
				new_messages[0] = SystemMessage(
					content=[ContentPartTextParam(type='text', text=billing), *first.content],
					cache=first.cache,
				)
		else:
			new_messages.insert(0, SystemMessage(content=billing))
		return new_messages

	async def ainvoke(self, messages: list[BaseMessage], output_format: Any = None, **kwargs: Any):  # type: ignore[override]
		return await super().ainvoke(self._inject_billing_header(messages), output_format=output_format, **kwargs)
