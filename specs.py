

import ps

def assert_quote(input, expected):
    quotes = [m.group() for m in ps.find_quoted_quotes(input)]
    assert expected == quotes, repr(quotes)

def assert_percent(input, expected):
    assert expected == ps.percent_quoted(input)

class TestPercentQuoted:
    
    def test_it_should_find_percentage_of_quoted_text(self):    
        assert_percent(
            'Hello. I said, "yes"',
            25,
            )

class TestCalcNumberOfCharacters:

    def test_it_should_count_the_number_of_characters(self):
        text = 'Hello. I said, "yes"'
        assert 20 == ps.calc_number_of_characters(text)

class TestCalcNumberOfQuotes:

    def test_it_should_count_the_number_of_quoted_characters(self):
        text = 'Hello. I said, "yes"'
        assert 5 == ps.calc_number_of_quotes(text)

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
