"""Prompt templates transcribed as closely as possible from Feng et al.
(2025) "TEaR: Improving LLM-based Machine Translation with Systematic
Self-Refinement", Appendix B (Tables 17-19: Translate, Estimate, Refine).

Condition B's refine prompt (QE_REFINE_TEMPLATE) adapts the paper's own
XCOMET-Score external-feedback baseline (Table 22) to COMET-KIWI — the
paper already tested exactly this "external scalar QE score instead of
self-critique" pattern, just with XCOMET rather than COMET-KIWI, so this
is a faithful substitution rather than an invented design.
"""

TRANSLATE_TEMPLATE = """Please provide the {tgt_lang} translation for the {src_lang} sentences:
Source: {source_text}
Target:
Output only the {tgt_lang} translation, with no explanation, label, or extra text."""

# Table 18's 3-shot MQM examples, verbatim.
ESTIMATE_TEMPLATE = """Please identify errors and assess the quality of the translation.
The categories of errors are accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.

Each error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technical errors but do not disrupt the flow or hinder comprehension.

Example1:
Chinese source: 大众点评乌鲁木齐家居商场频道为您提供居然之家地址，电话，营业时间等最新商户信息， 找装修公司，就上大众点评
English translation: Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.
MQM annotations:
critical: accuracy/addition - "of high-speed rail"
major: accuracy/mistranslation - "go to the reviews"
minor: style/awkward - "etc.,"

Example2:
English source: I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.
German translation: Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.
MQM annotations:
critical: no-error
major: accuracy/mistranslation - "involvement"
    accuracy/omission - "the account holder"
minor: fluency/grammar - "wäre"
    fluency/register - "dir"

Example3:
English source: Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.
Czech transation: Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemže obě partaje se snaží posoudit vyhlídky na úspech po posledních výměnách v jednáních.
MQM annotations:
critical: no-error
major: accuracy/addition - "ve Vídni"
    accuracy/omission - "the stop-start"
minor: terminology/inappropriate for context - "partake"

{src_lang} source: {source_text}
{tgt_lang} translation: {translation}
MQM annotations:"""

REFINE_TEMPLATE = """Please provide the {tgt_lang} translation for the {src_lang} sentences.
Source: {source_text}
Target: {translation}
I'm not satisfied with this target, because some defects exist: {feedback}
Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technical errors but do not disrupt the flow or hinder comprehension.
Upon reviewing the translation examples and error information, please proceed to compose the final {tgt_lang} translation to the sentence: {source_text}. First, based on the defects information locate the error span in the target segment, comprehend its nature, and rectify it. Then, imagine yourself as a native {tgt_lang} speaker, ensuring that the rectified target segment is not only precise but also faithful to the source segment.
Output only the final {tgt_lang} translation, with no explanation, label, or extra text."""

# Condition B: adapted from Table 22 (XCOMET-Score), substituting COMET-KIWI
# (reference-free, matching the Estimate module's original reference-free
# design) for XCOMET, and a scalar sentence-level score rather than the
# self-generated MQM span critique. Standard COMET-KIWI (wmt22-cometkiwi-da)
# only produces a sentence-level score, not error spans — see
# results/phase_4_summary.md for why we didn't force a span-level design.
QE_REFINE_TEMPLATE = """Please provide the {tgt_lang} translation for the {src_lang} sentences.
Source: {source_text}
Target: {translation}
Its COMET-KIWI quality score is {score:.3f} (out of 1.0), which is relatively low.
Please give me the final target translation that might have a higher COMET-KIWI score.
Output only the final {tgt_lang} translation, with no explanation, label, or extra text."""
