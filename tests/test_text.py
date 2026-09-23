from neuroclone.speech.text import (
    SentenceChunker,
    ThinkFilter,
    clean_for_tts,
    extract_tags,
    has_speakable_content,
    strip_speaker_prefix,
)


def chunk(parts, **kw):
    c = SentenceChunker(**kw)
    out = []
    for p in parts:
        out += c.feed(p)
    return out + c.flush()


def test_sentences_split_on_terminators_across_tokens():
    assert chunk(["Hello", " there", "! How", " are you", "? Fine."], first_clause_chars=0) == [
        "Hello there!", "How are you?", "Fine."]


def test_abbreviations_decimals_and_initials_do_not_split():
    out = chunk(["Dr. Smith paid 3.50 dollars. J. R. R. Tolkien wrote e.g. books. I did it, not I. Done"],
                first_clause_chars=0)
    assert out == ["Dr. Smith paid 3.50 dollars.", "J. R. R. Tolkien wrote e.g. books.", "I did it, not I.", "Done"]


def test_boundary_waits_for_next_char():
    c = SentenceChunker(first_clause_chars=0)
    assert c.feed("Wait.") == []  # could still be "Wait.com" or an abbreviation
    assert c.feed(" Next") == ["Wait."]


def test_early_first_clause_cuts_long_openers_only_once():
    text = "Okay so here is the thing about this entire situation, it is honestly wild. Next one, also long enough here."
    words = [w + " " for w in text.split(" ")]  # streamed word by word, like real tokens
    out = chunk(words, first_clause_chars=40)
    assert out[0] == "Okay so here is the thing about this entire situation,"
    assert out[1] == "it is honestly wild."
    assert out[2] == "Next one, also long enough here."
    # When a whole sentence is already available there is nothing to gain by cutting it.
    assert chunk([text], first_clause_chars=40)[0].endswith("wild.")


def test_newlines_and_max_length():
    assert chunk(["line one\nline two"], first_clause_chars=0) == ["line one", "line two"]
    long = "word " * 80
    pieces = chunk([long], first_clause_chars=0, max_chars=100)
    assert all(len(p) <= 100 for p in pieces) and len(pieces) > 1


def test_think_filter_handles_split_tags():
    tf = ThinkFilter()
    out = tf.feed("<thi") + tf.feed("nk>plan secretly</thi") + tf.feed("nk>Hi chat <3") + tf.flush()
    assert out == "Hi chat <3"
    tf = ThinkFilter()
    assert tf.feed("reasoning without open</think>Answer") + tf.flush() == "reasoning without openAnswer"
    tf = ThinkFilter()
    assert tf.feed("<thinking>x</thinking>ok") == "ok"


def test_extract_tags_and_actions():
    text, tags = extract_tags("[Smug] I *giggles* win [laughs] again [excited]")
    assert text == "I win again"
    assert tags == ["smug", "excited", "happy"]


def test_clean_for_tts():
    assert clean_for_tts('"**Hi** see https://example.com 😀 `x`"') == "Hi see a link x"
    assert clean_for_tts("- item one") == "item one"
    assert clean_for_tts("cool_user123 said hi") == "cool_user123 said hi"


def test_speaker_prefix_and_speakable():
    assert strip_speaker_prefix("Nexa: hello", ["Nexa"]) == "hello"
    assert strip_speaker_prefix("(Nexa): hi", ["Nexa"]) == "hi"
    assert strip_speaker_prefix("Nexagon: hi", ["Nexa"]) == "Nexagon: hi"
    assert has_speakable_content("ok!") and not has_speakable_content("...!?")
