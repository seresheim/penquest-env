import asyncio
import signal
from typing import (
    Optional,
    Dict,
    Tuple,
    Coroutine,
    AsyncGenerator,
    Any,
    Callable
)

from penquest_env.network.WebsocketConnector import WebsocketConnector

from penquest_pkgs.utils import get_logger, parse_stream, write_msg
from penquest_pkgs.game import InputEvents


# Constants
FIELD_TYPE = "type"
FIELD_DATA = "data"
FIELD_CONNECTION_ID = "connectionId"

SUBFIELD_EVENT = "event"
SUBFIELD_DATA = "data"

SUBSUBFILED_CON_ID = "connectionId"

class Timer:
    """Async Timer that calls a callback after a fixed amount of time has passed

    Attributes:
        _timeout(float): amount of seconds how long the timer shall wait until
            it calls the callback function
        _callback(Callable): function that is called after the timeout
        _active(bool): a flag indicating whether the timer is currently active
            or not
        _task(asyncio.Task): the async task that waits the time and calls the
            callback afterwards
    """

    def __init__(self, timeout: float, callback: Callable):
        """initializes attributes

        :param timeout: amount of seconds how long the timer shall wait until
            it calls the callback function
        :param callback: function that is called after the timeout
        """
        self._timeout = timeout
        self._callback = callback
        self._active = True
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        """Asynchronously sleeps for the amount of seconds specified in the 
        constructor and calls the callback function afterwards 
        """
        await asyncio.sleep(self._timeout)
        await self._callback()

    def is_active(self) -> bool:
        """Indicates whether the timer is currently active

        :return: active flage
        """
        return self._active

    def cancel(self):
        """Cancels the timer. The timer interrupts the waiting process. No
        callback function is called afterwards
        """
        if self._active:
            self._task.cancel()
            self._active = False


class SessionMiddleware:

    """Multiplexes messages from multiple environments over a single websocket 
    connection. 

    This class is executed in a separat process and communicates with
    other environment processes via a local socket.

    Attributes:
        host(str): host address of the PenQuest server
        port(int): port where the PenQuest server is listens on the host
        api_key(str): Identifies and authorizes the player to play PenQuest
            on the server
        _websocket(WebsocketConnector): handles the communication over the 
            websocket
        _envs(Dict[str, Tuple[int, asyncio.StreamReader, asyncio.StreamWriter]]):
            maps connection IDs to environment IDs, input_channel and 
            output_channel of environment processes
        _con_ids(Dict[int, str]): maps environment IDs to connection IDs
        _ws_input_channel(asyncio.Queue): input_channel that transports messages
            from the websockets to this class
        _ws output_channel(asyncio.Queue): output_channel that transports
            messages to the websocket
        _serving_coroutine(asyncio.Coroutine): listens for incoming socket 
            connections from environment processes and forwards them to a
            handler
        _idle_cancelation_timer(Timer): disconnects and shutsdown the websocket 
            connection/process after no environment process connected for a 
            certain timeout
    """

    _id_counter = 0

    def get_id() -> int:
        """Returns a unique identifier

        :return: unique id
        """
        SessionMiddleware._id_counter += 1
        return SessionMiddleware._id_counter

    # Constructor
    def __init__(
            self, 
            api_key: str, 
            host: str, 
            port: int,
            internal_port: int,
            timeout_con_start: int,
            timeout_con_restart: int
        ):
        self.api_key = api_key
        self.host = host
        self.port = port
        self.internal_port = internal_port
        self.timeout_con_start = timeout_con_start
        self.timeout_con_restart = timeout_con_restart
        arguments = {"host": host, "port": port}
        self._websocket = WebsocketConnector.get_connector(arguments)
        self._envs: Dict[str, Tuple[int, asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self._con_ids: Dict[int, str] = {}
        self._ws_input_channel: asyncio.Queue = None
        self._ws_output_channel: asyncio.Queue = None
        self._serving_coroutine: Coroutine = None
        self._idle_cancelation_timer: Timer = None

    def start(self): 
        """Installs a listener for process signaling (SIGTERM) and starts the
        listening coroutine on the local socket
        """
        async def _async_start():
            """Establishes the websocket connection and starts listening on the
            local socket for environments to forward messages.
            """
            if self._websocket.is_connected():
                return

            # setup websocket connection to backend
            await self._websocket.connect(self.api_key)
            ws_input_channel, ws_output_channel = self._websocket.get_channels()
            self._ws_input_channel = ws_input_channel
            self._ws_output_channel = ws_output_channel
            asyncio.create_task(
                self.unpack_incoming_messages(self._ws_input_channel)
            )
            get_logger(__name__).debug(
                "SessionMiddleware now listening for incoming websocket "
                "messages."
            )

            # setup socket connection to other processes
            server = await asyncio.start_server(
                self._client_connected, 
                'localhost', 
                self.internal_port
            )
            get_logger(__name__).debug(
                "SessionMiddleware now listening for incoming local socket "
                "messages."
            )

            self._idle_cancelation_timer = Timer(
                self.timeout_con_start, 
                self._check_for_connections
            )

            async with server:
                self._serving_coroutine = server.serve_forever()
                await self._serving_coroutine

        
        def _terminate_connection(sign_number: int, frame):
            """Cleanup the websocket connection after receiving a termination
            signal

            :param sign_number: number of signal
            :param frame: no clue
            """
            asyncio.run(self._close())

        signal.signal(signal.SIGTERM, _terminate_connection)
        asyncio.get_event_loop().run_until_complete(_async_start())

    async def _client_connected(
            self, 
            reader: asyncio.StreamReader, 
            writer: asyncio.StreamWriter
        ):
        """Registers a newly connected environment process that connected on the
        local socket and creates a new listening task on its reader.

        :param reader: input_channel of the socket connection to the environment
            process
        :param writer: output_channel of the socket connection to the 
            environment process
        """
        if self._websocket is None or not self._websocket.is_connected():
            return
        env_id = SessionMiddleware.get_id()

        encoded_msg = await reader.readline()
        connection_id = encoded_msg.decode('utf-8').rstrip("\n")
        self._envs[connection_id] = (env_id, reader, writer)
        self._con_ids[env_id] = connection_id
        
        # Stop Idle-Cancelation Timer
        if self._idle_cancelation_timer.is_active():
            self._idle_cancelation_timer.cancel()
        get_logger(__name__).info(
            f"Connected env({env_id}) '{connection_id}' in the ws process"
        )
        asyncio.create_task(self._handle_messages(env_id, reader))

    async def _check_for_connections(self):
        """Shuts down the websocket connection if no environment process is
        currently connected to this process.
        """
        if len(self._envs) == 0:
            get_logger(__name__).info(
                "Websocket process shuting down due to no environment"
                f"connecting to it"
            )
            await self._close()

    async def _close(self):
        """Cleans up the task listening on the websocket and signals the
        websocket connection to disconnect gracefully
        """
        if self._ws_output_channel is not None and self._websocket.is_connected():
            await self._ws_output_channel.put(None)

        if self._serving_coroutine is not None:
            self._serving_coroutine.close()
            self._serving_coroutine = None

    async def _handle_messages(self, env_id:int, reader: asyncio.StreamReader):
        """Forwards incoming messages from the environment process to the
        websocket

        :param env_id: id of the environment 
        :param reader: input_channel of the environment
        """
        get_logger(__name__).debug(
                f"Start handling messages for env({env_id}) after "
                f"connection established"
            )
        async for msg_type, msg in parse_stream(reader):
            get_logger(__name__).debug(
                f"Received a message for env({env_id}) in SessionMiddleware"
            )
            if msg is None:
                break
            await self.pack_outgoing_messages(env_id, msg_type, msg)
        await self._close_connection(env_id)
    
    async def _close_connection(self, env_id: int):
        """Closes a connection to an environment process and unrgesiters it

        :param env_id: id of the environment 
        """
        if env_id in self._con_ids:
            connection_id = self._con_ids[env_id]
            if connection_id in self._envs:
                _, input_channel, output_channel = self._envs[connection_id]
                input_channel.feed_eof()
                output_channel.write_eof()
                output_channel.close()
                del self._envs[connection_id]
            del self._con_ids[env_id]
            get_logger(__name__).info(
                    f"connection to env {env_id} with connection_id "
                    f"{connection_id} in websocket process closed"
                )

        if len(self._envs) == 0:
            self._idle_cancelation_timer = Timer(
                self.timeout_con_restart, 
                self._check_for_connections
            )
    
    def _close_all_connections(self):
        """Closes the connections to all registered environment processes.
        This method is usually invoked after the websocket connection got 
        destroyed
        """
        for env_id, input_stream, output_stream in self._envs.values():
            input_stream.feed_eof()
            output_stream.write_eof()
            output_stream.close()
        if self._serving_coroutine is not None:
            self._serving_coroutine.close()
            self._serving_coroutine = None

    async def pack_outgoing_messages(
            self, 
            env_id: int, 
            msg_type: str,
            msg: Dict[str, Any]
        ):
        """Wraps outgoing messages in a dictionary including information about
        the environment the message came from 

        :param env_id: identifier of the environment
        :param msg_type: lowest level message type
        :param msg: the original message that is wrapped
        """
        connection_id = self._con_ids[env_id]
        msg = {
            FIELD_CONNECTION_ID: connection_id,
            FIELD_TYPE: msg_type,
            FIELD_DATA: msg
        }
        await self._ws_output_channel.put(msg)

    async def unpack_incoming_messages(self, input_channel: AsyncGenerator):
        """Unwrapps messages coming from the websocket and routes it to the 
        correct environment process

        :param input_channel: input_channel from the websocket
        :raises ValueError: required fields were not found
        """
        async for msg in input_channel:
            try:
                if msg is None: break

                # Check if message is valid
                required_fields = [FIELD_CONNECTION_ID, FIELD_DATA]
                not_found_fields = [it for it in required_fields if it not in msg]
                if len(not_found_fields) > 0:
                    raise ValueError(
                        f"message aborted: protocol error - missing required "
                        f"fields: {', '.join(not_found_fields)}"
                    )
                
                # Get required fields
                connection_id = msg[FIELD_CONNECTION_ID]
                data = msg[FIELD_DATA]

                # Spy on receiver for connection_id changes
                # the server sets the permanent connection ID for an environment
                # however the very first message uses a temporary connection ID,
                # which is also stored in the routing table. Therefore the
                # routing table needs to be updated and therefore the next
                # packet level needs to be unwrapped
                if SUBFIELD_EVENT in data and data[SUBFIELD_EVENT] == InputEvents.NEW_CONNECTION_ID:
                    if SUBFIELD_DATA in data:
                        sub_data = data[SUBFIELD_DATA]
                        if SUBSUBFILED_CON_ID in sub_data:
                            new_connection_id = sub_data[SUBSUBFILED_CON_ID]
                            old_connection_id = connection_id
                            env_id, stream_reader, stream_writer = self._envs[connection_id]
                            self._envs[new_connection_id] = env_id, stream_reader, stream_writer
                            self._con_ids[env_id] = new_connection_id
                            del self._envs[connection_id]
                            connection_id = new_connection_id
                            get_logger(__name__).info(
                                f"updated connection_id '{old_connection_id}' "
                                f"to '{new_connection_id}'"
                            )

                # Route data to the correct channel
                if connection_id in self._envs:
                    (_, input_stream, output_stream) = self._envs[connection_id]
                    await write_msg(data, output_stream)
                else:
                    get_logger(__name__).warning(
                        f"received message for unknon connection id: "
                        f"'{connection_id}', message: '{data}'"
                    )
            except Exception as e:
                get_logger(__name__).error(f"message aborted: {e}")
                # print Stacktrace
                import traceback
                traceback.print_exc()
                continue
        self._close_all_connections()
