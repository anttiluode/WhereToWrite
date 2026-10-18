"""Regression check: loading Qwen must enable the Ask button.

Uses fake widgets and an immediately completed worker; no Tk display or
transformer model is needed.
"""
from concurrent.futures import Future
from unittest import TestCase
from unittest.mock import Mock, call

from where_to_write.gui import Window


class CompletedPool:
    def submit(self, function):
        future = Future()
        future.set_result(function())
        return future


class GUIStateTests(TestCase):
    def test_ask_enabled_after_successful_model_load(self):
        window = object.__new__(Window)
        window.pool = CompletedPool()
        window.root = Mock()
        window.root.after.side_effect = lambda _delay, callback: callback()
        window.load_button = Mock()
        window.ask_button = Mock()
        window.status = Mock()
        window.model = None

        sentinel = object()
        window._future(lambda: sentinel, lambda result: setattr(window, 'model', result))

        self.assertIs(window.model, sentinel)
        self.assertFalse(window.running)
        self.assertEqual(window.ask_button.configure.call_args, call(state='normal'))
        self.assertEqual(window.load_button.configure.call_args, call(state='normal'))
