"""Shutdown checks for the local HiPerGator tunnel."""

import unittest
from unittest.mock import patch

from src.local.utils.tunnel import TunnelManager


class Gateway:
    def __init__(self, events):
        self.events = events

    def is_active(self):
        return True

    def close(self):
        self.events.append("close")


class TunnelShutdownTests(unittest.TestCase):
    def test_stop_cancels_tracked_job_before_closing_ssh(self):
        events = []
        manager = TunnelManager("example.org", "user", "/repo")
        manager._gateway = Gateway(events)
        manager._remember_job("12345")
        with patch("src.local.utils.tunnel.remote_command",
                   side_effect=lambda gateway, command: events.append(command)):
            manager.stop(cancel_job=True)
        self.assertEqual(events, ["scancel 12345", "close"])

    def test_stop_without_job_closes_ssh(self):
        events = []
        manager = TunnelManager("example.org", "user", "/repo")
        manager._gateway = Gateway(events)
        manager._connection_finished = True
        with patch("src.local.utils.tunnel.remote_command") as command:
            manager.stop(cancel_job=True)
        command.assert_not_called()
        self.assertEqual(events, ["close"])


if __name__ == "__main__":
    unittest.main()
