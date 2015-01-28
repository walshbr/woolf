

import ps


def assert_quote(input, expected):
    quotes = [m.group() for m in ps.find_quoted_quotes(input)]
    assert expected == quotes, repr(quotes)


class TestFindQuotedQuotes:

    def test_it_should_find_double_quotes(self):
        assert_quote(
            'She said, "Howdy!" ' * 100,
            ['"Howdy!"'] * 100,
            )

    def test_it_should_find_single_quotes(self):
        assert_quote(
            "She said, 'Howdy!'",
            ["'Howdy!'"],
            )

    def test_it_should_find_quotes_with_preceding_contractions(self):
        assert_quote(
            "She didn't say, 'Good-bye.'",
            ["'Good-bye.'"],
            )

    def test_it_should_find_quotes_containing_contractions(self):
        assert_quote(
            "She said, 'Don't say that!'",
            ["'Don't say that!'"],
            )

    def test_it_should_find_quotes_with_contractions_following(self):
        assert_quote(
            "She said, 'Say that!' Or she didn't.",
            ["'Say that!'"],
            )

    def test_it_should_find_quotes_surrounded_by_contractions(self):
        assert_quote(
            "She didn't said, 'Don't say that!' Or she didn't.",
            ["'Don't say that!'"],
            )

    def test_it_should_find_multiple_quotes_containing_contractions(self):
        assert_quote(
            "She didn't say, 'Don't say that!' "
            "She didn't say, 'Don't say that!' Something else here.",
            ["'Don't say that!'", "'Don't say that!'"],
            )
