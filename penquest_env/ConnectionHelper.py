from penquest_pkgs.utils import retrieve_value_from_config, get_logger
from penquest_pkgs.game import Game, GameInputInterpreter, GameOutputInterpreter

from typing import Tuple

import asyncio
import time
import configparser
import uuid

CONFIG_FIELD_INTERNAL = "internal"
CONFIG_FIELD_PORT = "port"

class ConnectionHelper:

    def __init__(self, game: Game):
        self.game = game

    async def _try_connect_process(
            self,
            port: int
        ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Tries to connect to the process that handles the websocket connection
        with the PenQuest server across all environments.

        :param port: over which the environment communicates
            with the websocket process
        :raises ConnectionRefusedError: if max amount of tries exceeded
        :return: tuple with of input stream and output stream to the websocket
            process
        """
        # changeable from config file
        input_stream = None
        output_stream = None
        max_amount_tries = 5
        delay = 0.5
        tries = 0

        while tries < max_amount_tries and input_stream is None and output_stream is None:
            try:
                input_stream, output_stream  = await asyncio.open_connection(
                    'localhost', 
                    port
                )
            except ConnectionRefusedError:
                tries += 1
                time.sleep(delay)
        if tries == max_amount_tries and input_stream is None and output_stream is None:
            raise ConnectionRefusedError(
                f"Cannot connect to other process on port {port}"
            )
        
        return input_stream, output_stream

    async def connect_to_server(self, config_file_path :str):
        """Initiates a connection for the current environment with the 
        subprocess that handles the websocket connection

        :param config_file_path: file path to a configuarion file that stores
            stores the internal port over which the environment communicates
            with the websocket process
        """
        get_logger(__name__).debug("Connect env to server ...")

        config = configparser.ConfigParser()
        config.read(config_file_path)
        port = retrieve_value_from_config(
            config, 
            CONFIG_FIELD_INTERNAL, 
            CONFIG_FIELD_PORT, 
            int,
            "internal port",
        )

        input_stream, output_stream = await self._try_connect_process(port)
        temporary_connection_id = f"env-{uuid.uuid4()}"

        # send temporary connection id to ws process
        payload = f"{str(temporary_connection_id)}\n"
        encoded_payload = bytes(payload, 'utf-8')
        output_stream.writelines([encoded_payload])
        await output_stream.drain()

        self.in_interpreter = GameInputInterpreter(input_stream)
        self.out_interpreter = GameOutputInterpreter(output_stream)
        await self.in_interpreter.start_listening_job(self.game)
        await self.out_interpreter.start_listening_to_game_events(self.game)
        await self.game.request_connection_id()
        get_logger(__name__).debug(
            f"Established connection with connection_id "
            f"'{self.game.actor_connection_id}'"
        )