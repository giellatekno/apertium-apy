"""/translateLLM: like /translate, but the Apertium output is post-edited by
an LLM to read more idiomatically.  Takes the same arguments as /translate.

Response:

    {"responseData": {
        "translatedText": <post-edited text, or the Apertium text if llmStatus != "ok">,
        "apertiumText":   <raw Apertium output, honouring markUnknown>,
        "llmStatus":      "ok" | "filtered" | "unavailable" | "too_long",
        "llmModel":       <model that answered, or null>},
     "responseDetails": null, "responseStatus": 200}
"""

import asyncio
import logging
import re

import tornado.iostream
from tornado import gen

from apertium_apy.handlers.translate import TranslateHandler
from apertium_apy.utils import llm_postedit
from typing import Optional  # noqa: F401


class TranslateLLMHandler(TranslateHandler):
    llm_config = None  # type: Optional[llm_postedit.LLMConfig]  # set by llm_handlers()

    @gen.coroutine
    def translate_and_respond(self, pair, pipeline, to_translate, mark_unknown, nosplit=False, deformat=True, reformat=True, prefs=''):
        mark_unknown = mark_unknown in ['yes', 'true', '1']
        self.note_pair_usage(pair)
        before = self.log_before_translation()
        try:
            translated = yield pipeline.translate(to_translate, nosplit, deformat, reformat, prefs)
        except (asyncio.TimeoutError, tornado.iostream.StreamClosedError) as e:
            logging.warning('Translation error in pair %s-%s: %s', pair[0], pair[1], e)
            pipeline.stuck = True
            self.send_error(503, explanation='internal error')
            self.clean_pairs()
            return
        self.log_after_translation(before, len(to_translate))
        self.clean_pairs()

        apertium_text = self.maybe_strip_marks(mark_unknown, pair, translated)
        # The LLM never sees unknown-word marks, it would try to "fix" them
        llm_input = re.sub(self.unknown_mark_re, r'\1', translated)
        result = yield llm_postedit.post_edit(self.llm_config, llm_input, pair[0], pair[1])
        logging.info('LLM post-edit %s-%s: %s (model %s)', pair[0], pair[1], result.status, result.model)

        self.send_response({
            'responseData': {
                'translatedText': result.text if result.status == llm_postedit.OK else apertium_text,
                'apertiumText': apertium_text,
                'llmStatus': result.status,
                'llmModel': result.model,
            },
            'responseDetails': None,
            'responseStatus': 200,
        })


def llm_handlers(config_path):
    """Returns the route list to add to the application; empty unless the
    config file has a usable [LLM] section."""
    if not config_path:
        return []
    config = llm_postedit.LLMConfig.from_config_file(config_path)
    if config is None:
        return []
    TranslateLLMHandler.llm_config = config
    logging.info('LLM post-editing enabled on /translateLLM (model %r)', config.model)
    return [(r'/translateLLM', TranslateLLMHandler)]
