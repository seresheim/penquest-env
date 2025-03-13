import multiprocessing

from penquest_env.network.SessionMiddleware import SessionMiddleware
from penquest_env.utils.config_loader import Config, ConfigLoader


def start(
        api_key: str=None,
        host: str=None,
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
    config = ConfigLoader.load_config(config_path=config_file_path)
    if api_key is not None:
        config.api_key = api_key
    if host is not None:
        config.host = host
    process = multiprocessing.Process(
        target=_start,
        args=(config,)
    )
    process.start()

def _start(config: Config):
    """Initial method of the websocket connection process

    :param config: a Config object that contains all necessary information
    """
    session = SessionMiddleware(config)
    session.start()
