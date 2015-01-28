

import ps


class TestFindQuotedQuotes:

    def test_it_should_find_double_quotes(self):
        quotes = [
            m.group() for m in ps.find_quoted_quotes('She said, "Howdy!"')
            ]
        assert quotes == ["Howdy!"]
