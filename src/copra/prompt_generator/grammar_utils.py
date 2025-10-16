#!/usr/bin/env python3

"""
Grammar Utilities Module

This module provides picklable utility classes for grammar parsing,
particularly for Python 3.14t multiprocessing compatibility.
"""


class StringParser:
    """
    Picklable string parser class for grammar parsing.

    This parser extracts string tokens from text by finding content
    between keyword markers. It's designed to be picklable for use
    with Python 3.14t's forkserver multiprocessing.

    Attributes:
        keywords: List of keyword strings to recognize as delimiters
    """

    def __init__(self, keywords):
        """
        Initialize the string parser.

        Args:
            keywords: List of keyword strings that mark token boundaries
        """
        self.keywords = keywords

    def __call__(self, text, pos):
        """
        Parse string tokens from text starting at position pos.

        Scans through the text looking for backtick characters ('`'),
        then checks if any keyword starts at that position. Returns
        the text content before the first matching keyword.

        Args:
            text: The text to parse
            pos: Starting position in the text

        Returns:
            The string content before the first keyword, or None if
            no keyword is found
        """
        last = pos
        while last < len(text):
            while last < len(text) and text[last] != '`':
                last += 1
            if last < len(text):
                for keyword in self.keywords:
                    if text[last:].startswith(keyword):
                        return text[pos:last]
                last += 1
        return None
