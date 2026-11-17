"""Post-editing of Apertium output through an OpenAI-compatible chat
completions endpoint (e.g. Azure OpenAI behind an API gateway).

Configured from an [LLM] section in the APy config file (-C), see
llm.example.conf.  The API key is read from the environment variable
APY_LLM_API_KEY, or from `api_key` in the config file as a fallback.
"""

import configparser
import json
import logging
import os

from tornado import httpclient

API_KEY_ENV = 'APY_LLM_API_KEY'

# Result statuses, sent to the client as `llmStatus`:
OK = 'ok'                    # translatedText is the post-edited text
FILTERED = 'filtered'        # the provider refused (content filter / refusal)
UNAVAILABLE = 'unavailable'  # network error, timeout, HTTP error, bad or truncated response
TOO_LONG = 'too_long'        # input exceeded max_chars, the LLM was not called

DEFAULT_PROMPT = (
    'You are a careful post-editor of machine translation. The text you receive was '
    'translated by a rule-based machine translation system into the language with '
    'ISO 639-3 code "{target_lang}" (from "{source_lang}"). Its meaning is generally '
    'accurate, but the phrasing can be unidiomatic or stilted. Rewrite it so it reads '
    'naturally and idiomatically in "{target_lang}", while keeping the meaning, '
    'level of detail, names, numbers and formatting (line breaks, punctuation style) '
    'exactly as they are. Do not add, remove or explain anything. Leave words you do '
    'not recognise unchanged. Reply with the rewritten text only.'
)


class LLMResult:
    def __init__(self, status, text=None, model=None):
        self.status = status
        self.text = text
        self.model = model

    def __repr__(self):
        return 'LLMResult(status={!r}, model={!r})'.format(self.status, self.model)


class LLMConfig:
    def __init__(self, endpoint, api_key, api_key_header='api-key', model='',
                 include_model_in_body=False, temperature=0.15, timeout=30.0,
                 max_chars=10000, prompt=DEFAULT_PROMPT):
        self.endpoint = endpoint.replace('{model}', model)
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.model = model
        self.include_model_in_body = include_model_in_body
        self.temperature = temperature
        self.timeout = timeout
        self.max_chars = max_chars
        self.prompt = prompt

    def system_prompt(self, source_lang, target_lang):
        # str.replace rather than str.format, so prompts may contain other braces
        return self.prompt.replace('{source_lang}', source_lang).replace('{target_lang}', target_lang)

    @classmethod
    def from_config_file(cls, path, section='LLM'):
        """Returns an LLMConfig, or None if the file has no such section or it is incomplete."""
        conf = configparser.ConfigParser(interpolation=None)  # prompts may contain %
        conf.read(path)
        if section not in conf:
            return None
        sect = conf[section]
        if not sect.getboolean('enabled', fallback=True):
            logging.info('LLM post-editing disabled in config')
            return None

        endpoint = sect.get('endpoint', '').strip()
        api_key = os.environ.get(API_KEY_ENV) or sect.get('api_key', '').strip()
        if not endpoint:
            logging.error('[%s] section in %s has no endpoint, LLM post-editing disabled', section, path)
            return None
        if not api_key:
            logging.error('No API key in $%s or [%s] api_key, LLM post-editing disabled', API_KEY_ENV, section)
            return None

        prompt = sect.get('prompt', '').strip()
        prompt_file = sect.get('prompt_file', '').strip()
        if prompt_file:
            if not os.path.isabs(prompt_file):
                prompt_file = os.path.join(os.path.dirname(os.path.abspath(path)), prompt_file)
            with open(prompt_file, encoding='utf-8') as f:
                prompt = f.read().strip()

        return cls(
            endpoint=endpoint,
            api_key=api_key,
            api_key_header=sect.get('api_key_header', 'api-key').strip(),
            model=sect.get('model', '').strip(),
            include_model_in_body=sect.getboolean('include_model_in_body', fallback=False),
            temperature=sect.getfloat('temperature', fallback=0.15),
            timeout=sect.getfloat('timeout', fallback=30.0),
            max_chars=sect.getint('max_chars', fallback=10000),
            prompt=prompt or DEFAULT_PROMPT,
        )


def _is_content_filter_error(body):
    try:
        error = json.loads(body).get('error') or {}
    except (ValueError, AttributeError):
        return False
    inner = error.get('innererror') or {}
    return (error.get('code') == 'content_filter' or
            inner.get('code') == 'ResponsibleAIPolicyViolation')


def parse_completion(body):
    """Maps a chat completions response body to an LLMResult."""
    try:
        data = json.loads(body)
        choice = data['choices'][0]
        message = choice.get('message') or {}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        logging.warning('LLM endpoint returned an unparseable response')
        return LLMResult(UNAVAILABLE)

    model = data.get('model')
    finish_reason = choice.get('finish_reason')
    if finish_reason == 'content_filter' or message.get('refusal'):
        return LLMResult(FILTERED, model=model)
    if finish_reason == 'length':
        logging.warning('LLM response was truncated (finish_reason=length)')
        return LLMResult(UNAVAILABLE, model=model)
    content = message.get('content')
    if not content or not content.strip():
        return LLMResult(UNAVAILABLE, model=model)
    return LLMResult(OK, text=content, model=model)


def _keep_outer_whitespace(original, edited):
    """LLMs strip leading/trailing whitespace; put the original's back."""
    lead = original[:len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    return lead + edited.strip() + trail


async def post_edit(config, text, source_lang, target_lang):
    """Sends Apertium output to the LLM. Never raises; returns an LLMResult."""
    if not text.strip():
        return LLMResult(OK, text=text)
    if len(text) > config.max_chars:
        return LLMResult(TOO_LONG)

    body = {
        'messages': [
            {'role': 'system', 'content': config.system_prompt(source_lang, target_lang)},
            {'role': 'user', 'content': text},
        ],
        'temperature': config.temperature,
        'stream': False,
    }
    if config.include_model_in_body and config.model:
        body['model'] = config.model

    request = httpclient.HTTPRequest(
        config.endpoint,
        method='POST',
        headers={'Content-Type': 'application/json', config.api_key_header: config.api_key},
        body=json.dumps(body),
        connect_timeout=min(config.timeout, 10.0),
        request_timeout=config.timeout,
    )
    try:
        response = await httpclient.AsyncHTTPClient().fetch(request)
    except httpclient.HTTPError as e:
        error_body = e.response.body if e.response is not None else b''
        if e.code == 400 and _is_content_filter_error(error_body):
            return LLMResult(FILTERED)
        logging.warning('LLM endpoint failed: HTTP %s %s', e.code, e.message)
        return LLMResult(UNAVAILABLE)
    except Exception as e:  # connection refused, DNS failure, SSL errors, …
        logging.warning('LLM endpoint failed: %s: %s', type(e).__name__, e)
        return LLMResult(UNAVAILABLE)

    result = parse_completion(response.body)
    if result.status == OK:
        result.text = _keep_outer_whitespace(text, result.text)
    return result
