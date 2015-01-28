

import ps


class TestFindQuotedQuotes:

    def test_it_should_find_double_quotes(self):
        quotes = [
            m.group()
            for m in ps.find_quoted_quotes('She said, "Howdy!" ' * 100)
            ]
        assert quotes == (['"Howdy!"'] * 100), repr(quotes)

    def test_it_should_find_single_quotes(self):
        quotes = [
            m.group()
            for m in ps.find_quoted_quotes("She said, 'Howdy!'")
            ]
        assert quotes == ["'Howdy!'"], repr(quotes)

    def test_it_should_find_quotes_with_preceding_contractions(self):
        quotes = [
            m.group()
            for m in ps.find_quoted_quotes("She didn't say, 'Good-bye.'")
            ]
        assert quotes == ["'Good-bye.'"], repr(quotes)

    def test_it_should_find_quotes_containing_contractions(self):
        quotes = [
            m.group()
            for m in ps.find_quoted_quotes("She said, 'Don't say that!'")
            ]
        assert quotes == ["'Don't say that!'"], repr(quotes)

    def test_it_should_find_quotes_with_contractions_following(self):
        quotes = [
            m.group()
            for m in ps.find_quoted_quotes("""
                She said, 'Say that!' Or she didn't.
                """)
            ]
        assert quotes == ["'Say that!'"], repr(quotes)

    def test_it_should_find_quotes_surrounded_by_contractions(self):
        quotes = [
            m.group()
            for m in ps.find_quoted_quotes("""
                She didn't said, 'Don't say that!' Or she didn't.
                """)
            ]
        assert quotes == ["'Don't say that!'"], repr(quotes)

    def test_it_should_find_multiple_quotes_containing_contractions(self):
        quotes = [
            m.group()
            for m in ps.find_quoted_quotes(
                "She didn't say, 'Don't say that!' "
                "She didn't say, 'Don't say that!' Something else here."
                )
            ]
        assert quotes == ["'Don't say that!'", "'Don't say that!'"], \
            repr(quotes)
