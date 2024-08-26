import os
import multiprocessing
import configparser

from penquest_env.network.SessionMiddleware import SessionMiddleware
from penquest_pkgs.utils import retrieve_value_from_config

DEFAULT_CONFIG_FILE = "default_config.ini"

FIELD_INTERNAL = "internal"
FIELD_EXTERNAL = "external"
FIELD_TIMEOUTS = "timeouts"
FIELD_API_KEY = "api_key"
FIELD_HOST = "host"
FIELD_PORT = "port"
FIELD_CON_START = "connection_start"
FIELD_CON_RESTART = "connection_restart"


def start(
        api_key: str=None, 
        host: str=None, 
        port: int=None, 
        config_file_path: str=None
    ):
    """Starts a separate process that handles the websocket connection to the 
    PenQuest server. In case multiple virtual environments of PenQuest are used,
    this process multiplexes the communication of all environments over a single
    websocket connection.

    :param api_key: API key to authorize the client. If you were not provided
        with an API key yet, please contact the developers at 
        https://www.pen.quest/
    :param host: host address of the PenQuest server to connect to. If this
        value is None (default), then a value in the default config file is 
        chosen. 
    :param port: host port of the PenQuest server to connect to. If this
        value is None (default), then a value in the default config file is 
        chosen. 
    :param config_file_path: file path to a configuration file. For more 
        information see the 'Configuration Files' section in the documentation.
        Defaults to a default configuration file within the package called 
        'default_config.ini'
    """
    if config_file_path is None:
        full_path = os.path.dirname(os.path.abspath(__file__))
        full_path = full_path.replace("/penquest_env/network", "/")
        config_file_path = os.path.join(full_path, DEFAULT_CONFIG_FILE)

    config = configparser.ConfigParser()
    config.read(config_file_path)
    api_key = retrieve_value_from_config(
        config, 
        FIELD_EXTERNAL, 
        FIELD_API_KEY, 
        str,
        "API key",
        parameter=api_key
    )
    external_host = retrieve_value_from_config(
        config, 
        FIELD_EXTERNAL, 
        FIELD_HOST, 
        str,
        "host",
        parameter=host
    )
    external_port = retrieve_value_from_config(
        config, 
        FIELD_EXTERNAL, 
        FIELD_PORT, 
        int,
        "port",
        parameter=port
    )
    internal_port = retrieve_value_from_config(
        config, 
        FIELD_INTERNAL, 
        FIELD_PORT, 
        int,
        "internal port",
    )
    timeout_con_start = retrieve_value_from_config(
        config, 
        FIELD_TIMEOUTS, 
        FIELD_CON_START, 
        int,
        "timeout connection start",
    )
    timeout_con_restart = retrieve_value_from_config(
        config, 
        FIELD_TIMEOUTS, 
        FIELD_CON_RESTART, 
        int,
        "timeout connection restart",
    )
    
    process = multiprocessing.Process(
        target=_start, 
        args=(
            api_key, 
            external_host, 
            external_port, 
            internal_port, 
            timeout_con_start, 
            timeout_con_restart
        )
    )
    process.start()

def _start(
        api_key: str, 
        host: str, 
        port: int, 
        internal_port: int, 
        timeout_con_start: int, 
        timeout_con_restart: int
    ):
    """Initial method of the websocket connection process

    :param api_key: API key to authorize the client
    :param host: host address of the PenQuest server to connect to. If this
        value is None, defaults to the default config file. Defaults to 
        None.
    :param port: host port of the PenQuest server to connect to. If this
        value is None, defaults to the default config file. Defaults to 
        None.
    :param internal_port: port that is used on the client machine to communicate
        between the differen environment processes and the websocket process
    :param timeout_con_start: timeout how long to wait for an environment 
        process to connect at start
    :param timeout_con_restart: timeout how long to wait for an environment 
        process to connect after a game has ended
    """
    session = SessionMiddleware(
        api_key, 
        host, 
        port, 
        internal_port, 
        timeout_con_start, 
        timeout_con_restart
    )
    session.start()
