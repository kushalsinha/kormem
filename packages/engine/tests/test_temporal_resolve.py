from datetime import datetime

from kormem.retrieve.temporal import describeOffset, resolveRelativeDates

SUNDAY = datetime(2023, 5, 28, 9, 52)


def testResolvesCommonPhrases():
    out = resolveRelativeDates(
        "[user]: Yesterday I ran; last Tuesday too, 3 weeks ago I started, "
        "and I fly next Friday.", SUNDAY,
    )
    assert "Yesterday [2023-05-27]" in out
    assert "last Tuesday [2023-05-23]" in out
    assert "3 weeks ago [2023-05-07]" in out
    assert "next Friday [2023-06-02]" in out


def testApproximatePhrasesAreMarked():
    out = resolveRelativeDates("[user]: I started it about a month ago.", SUNDAY)
    assert "about a month ago [≈2023-04-28]" in out
    assert "[≈" in resolveRelativeDates("[user]: a couple of days ago", SUNDAY)


def testLastWeekendFromAWeekend():
    assert "[2023-05-20/2023-05-21]" in resolveRelativeDates("[user]: last weekend", SUNDAY)
    assert "[2023-05-27/2023-05-28]" in resolveRelativeDates("[user]: last weekend", datetime(2023, 5, 29))


def testOnlyUserTurnsAndIdempotent():
    text = "[user]: two days ago\n[assistant]: A few years ago, studies showed yesterday's idea"
    once = resolveRelativeDates(text, SUNDAY)
    assert "[assistant]: A few years ago, studies showed yesterday's idea" in once
    assert resolveRelativeDates(once, SUNDAY) == once


def testNoAnchorIsNoop():
    assert resolveRelativeDates("[user]: yesterday", None) == "[user]: yesterday"


def testDescribeOffset():
    q = datetime(2023, 5, 30)
    assert describeOffset(datetime(2023, 5, 30), q) == "same day as the question"
    assert describeOffset(datetime(2023, 5, 27), q) == "3 days before the question"
    assert describeOffset(datetime(2023, 4, 30), q) == "4 weeks before the question"
