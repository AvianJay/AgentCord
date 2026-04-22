from __future__ import annotations

import unittest
from unittest.mock import patch

from agentcord import pterodactyl


class PterodactylListServersTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_servers_parses_root_listing_response(self) -> None:
        response = pterodactyl.PterodactylResponse(
            status=200,
            data={
                "data": [
                    {
                        "attributes": {
                            "identifier": "abc123",
                            "uuid": "uuid-1",
                            "name": "Lobby",
                            "description": "Main proxy",
                            "node": "Node A",
                            "server_owner": True,
                            "is_suspended": False,
                            "is_installing": False,
                            "current_state": "running",
                        }
                    },
                    {
                        "attributes": {
                            "identifier": "def456",
                            "uuid": "uuid-2",
                            "name": "Survival",
                        }
                    },
                ]
            },
            text="",
        )

        with patch.object(pterodactyl, "request_pterodactyl_client_api", return_value=response):
            result = await pterodactyl.list_pterodactyl_servers(None, None, None)

        self.assertEqual(
            result,
            [
                {
                    "identifier": "abc123",
                    "uuid": "uuid-1",
                    "name": "Lobby",
                    "description": "Main proxy",
                    "node": "Node A",
                    "is_owner": True,
                    "is_suspended": False,
                    "is_installing": False,
                    "current_state": "running",
                },
                {
                    "identifier": "def456",
                    "uuid": "uuid-2",
                    "name": "Survival",
                    "description": "",
                    "node": "",
                    "is_owner": False,
                    "is_suspended": False,
                    "is_installing": False,
                    "current_state": "",
                },
            ],
        )

    async def test_list_servers_rejects_invalid_response_shape(self) -> None:
        response = pterodactyl.PterodactylResponse(status=200, data={"meta": {}}, text="")

        with patch.object(pterodactyl, "request_pterodactyl_client_api", return_value=response):
            with self.assertRaisesRegex(pterodactyl.PterodactylError, "伺服器列表回應格式無效"):
                await pterodactyl.list_pterodactyl_servers(None, None, None)


if __name__ == "__main__":
    unittest.main()