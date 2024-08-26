import asyncio
import json
import websockets

from websockets import WebSocketClientProtocol
from typing import Any, Dict, AsyncGenerator, Tuple

from penquest_pkgs.utils import get_logger, EnumEncoder

FIELD_AUTHORIZATION_HEADER = "Authorization"

MSG_HELLO = "hello"
MSG_DATA = "data"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5001

async def stream_queue(queue: asyncio.Queue):
    while True:
        msg = await queue.get()
        yield msg
        if msg is None: break

class WebsocketConnector():
    """Handles the websocket connection from the environment to the PenQuest
    server. A single connection is used per client host, even if multiple
    environments are used concurrently.

    Attributes:
        host(str): host address of the PenQuest server
        port(int): port where the PenQuest server is listens on the host
        _send_queue(asyncio.Queue): input_channel that receives messages which
            are forwarded to the websocket
        _message_queue(asyncio.Queue): output_channel that forwards the messages
            coming from the websocket
        _connection(WebsocketClientProtocol): the websocket connection
        _listening_task(asyncio.Task): task that waits for messages from the 
            websocket and forwards them to the output_channel
        _sending_task(asyncio.Task): task that waits for messages on the 
            input_channel and forwards them to the websocket
    """

    @staticmethod
    def get_connector(connection_args: Dict[str, Any]) -> 'WebsocketConnector':
        """Creates a new WebsocketConnector instance with the provided 
        connection arguments

        :param connection_args: the connection arguments
        :return: the WebsocketConnector
        """
        default_values = {
            'host': DEFAULT_HOST,
            'port': DEFAULT_PORT,
        }

        connection_args = {**default_values, **connection_args}
        return WebsocketConnector(**connection_args)

    def __init__(
            self,
            host: str = DEFAULT_HOST,
            port: int = DEFAULT_PORT
        ):
        """Initializes all attributes.

        :param host: host address of the PenQuest server, defaults to 
            DEFAULT_HOST
        :param port: port where the PenQuest server is listens on the host, 
            defaults to DEFAULT_PORT
        """
        self.host = host if host is not None else DEFAULT_HOST
        self.port = port if port is not None else DEFAULT_PORT
        self._send_queue = asyncio.Queue()
        self._message_queue = asyncio.Queue()
        self._connection: WebSocketClientProtocol = None
        self._listening_task: asyncio.Task = None
        self._sending_task: asyncio.Task = None
        
    
    def get_channels(self) -> Tuple[AsyncGenerator, asyncio.Queue]:
        """Returns the input and output communication channels to other callers.
        The connector takes messages from the input channel, packs it 
        correspondingly and forwards it to the websocket. Also it forwards all
        incoming message from the websocket connection to the output channel
        after it unpacked it correspondingly.

        :return: output_channel, input_channel
        """
        return stream_queue(self._message_queue), self._send_queue

    async def connect(self, api_key: str):
        """Creates a websocket connection to the PenQuest server

        :param api_key: key that is used to authenticate the user
        """
        uri = f"ws://{self.host}:{self.port}/ws"
        self._connection = await websockets.connect(
            uri, 
            extra_headers={FIELD_AUTHORIZATION_HEADER: api_key}
        )
        # Start listening to incoming messages and sending outgoing messages
        await self._start_tasks()

    def is_connected(self) -> bool:
        """Determines whether the connector is currently connected via a
        websocket connection

        :return: True if connected, False otherwise
        """
        return self._connection is not None and self._connection.open

    async def _start_tasks(self):
        """Starts the listening tasks, which forward messages from the 
        input_channel to the websocket connection and from the websocket 
        connection to the output_channel.
        """
        self.listening_task = asyncio.create_task(self._receiveing_routine())
        self.sending_task = asyncio.create_task(self._sending_routine())

    async def disconnect(self):
        """Disconnects the current websocket connection if connected."""
        if self._connection is not None and self._connection.open:
            await self._connection.close()

    async def _disconnected(self):
        """Ends the forwarding of messages from the websocket connection to the
        input- and output_channel and vice versa.
        """
        if self._send_queue is not None:
            await self._send_queue.put(None)
            self._send_queue = None
        if self._message_queue is not None:
            await self._message_queue.put(None)
            self._message_queue = None
        if self._listening_task is not None:
            await self._listening_task
        if self._sending_task is not None:
            await self._sending_task
        if self._connection.closed:
            get_logger(__name__).info(
                f"Websocket connection to {self.host}:{self.port} closed"
            )

    async def _receiveing_routine(self):
        """Forwards messages from the websocket connection to the output_channel

        Requires an established websocket connection, which is built via 
        connect().

        :raises RuntimeError: websocket not connected
        """
        if self._connection is None:
            raise RuntimeError("Websocket not connected")
        
        try:
            async for message in self._connection:
                await self._handle_message(message)
        except websockets.ConnectionClosedError as e:
            if e.code !=  1000: # not OK
                get_logger(__name__).warn(
                    f"Connection closed abruptly: {str(e)}; code: {e.code}"
                )
        finally:
            await self._disconnected()

    async def _handle_message(self, message: str):
        """Parses incoming messages from the websocket connection via the json
        format and then forwards it to the output_channel

        :param message: message coming from the websocket connection
        """
        try:
            get_logger(__name__).debug(f"Received '{message}' over websocket")
            if message is None:
                raise ValueError("Empty message")
            parsed_message = json.loads(message)
            await self._message_queue.put(parsed_message)
        except Exception as e:
            get_logger(__name__).error(f"{str(e)}. Message aborted!")

    async def _sending_routine(self):
        """Encodes messages from the input_channel to strings in json format and
        forwards them  to the websocket connection.
        """
        while True:
            message = await self._send_queue.get()
            if message is None: break
            if not isinstance(message, str):
                message = json.dumps(message, cls=EnumEncoder)
            get_logger(__name__).debug(f"Sent '{message}' over websocket")
            await self._connection.send(message)
        if self._connection.open:
            await self.disconnect()