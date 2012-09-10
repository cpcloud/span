#!/usr/bin/env python

"""
"""

import os
import readline


def get_history_items():
    """
    """
    num_items = readline.get_current_history_length() + 1
    return [readline.get_history_item(i) for i in range(1, num_items)]


class HistoryCompleter(object):
    """
    """
    def __init__(self):
        """
        """
        self.__matches = []

    def complete(self, text, state):
        """
        """
        response = None
        if not state:
            history_values = get_history_items()
            if text:
                self.__matches = sorted(h for h in history_values
                                        if h and h.startswith(text))
            else:
                self.__matches = []
        try:
            response = self.matches[state]
        except IndexError:
            response = None
        return response

    @property
    def matches(self):
        return self.__matches


def input_loop(prompt='Prompt', converter=lambda x: x,
               history_filename='.completer.hist', default=None):
    """ """
    if os.path.exists(history_filename):
        readline.read_history_file(history_filename)
    try:
        while True:
            raw = eval(input(prompt))
            if not raw:
                break
            try:
                line = converter(raw)
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except ValueError:
                print(("""Invalid input value: "{0}" with type {1},
                need value of type {2}.""".format(raw, type(raw),
                                                 type(default))))
                continue
        else:
            line = default
    finally:
        readline.write_history_file(history_filename)
    return line


def setup_readline():
    """ """
    readline.set_completer(HistoryCompleter().complete)
    readline.parse_and_bind('tab: complete')


if __name__ == '__main__':
    setup_readline()
    input_loop()
