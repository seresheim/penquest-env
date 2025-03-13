import os
import configparser

from dataclasses import dataclass

DEFAULT_CONFIG_FILE = "default_config.ini"

FIELD_INTERNAL = "internal"
FIELD_EXTERNAL = "external"
FIELD_TIMEOUTS = "timeouts"
FIELD_API_KEY = "api_key"
FIELD_HOST = "host"
FIELD_PORT = "port"
FIELD_CON_START = "connection_start"
FIELD_CON_RESTART = "connection_restart"

@dataclass
class Config():
    api_key: str
    internal_port: int
    host: str
    timeout_connection_start: int
    timeout_connection_restart: int


class ConfigLoader():

    @staticmethod
    def get_config_file() -> str:
        """Looks in two locations whether a default_config.ini file is
        present. First it looks in the installed folder of the penquest-env
        package, second it looks in the current working directory. If the
        file is found it returns the path to the file.
        If not, it raises a FileNotFoundError.

        Raises:
            FileNotFoundError: default_config.ini file not found

        Returns:
            str: path to the config file
        """
        full_path = os.path.dirname(os.path.abspath(__file__))
        full_path = full_path.replace(
            f"{os.path.sep}penquest_env{os.path.sep}penquest_env",
            f"{os.path.sep}penquest_env"
        )
        config_file_path = os.path.join(full_path, DEFAULT_CONFIG_FILE)
        if os.path.exists(config_file_path):
            return config_file_path
        
        # in case the config file was not installed into the default
        # location the user has to create one
        config_file_path = os.path.join(os.getcwd(), DEFAULT_CONFIG_FILE)
        if os.path.exists(config_file_path):
            return config_file_path
        
        raise FileNotFoundError(
            f"Config file {DEFAULT_CONFIG_FILE} not found in "
            f"{full_path}. You need to create a default_config.ini "
            "file first."
        )
            

    @staticmethod
    def load_config(config_path: str = None) -> Config:
        """_summary_

        Args:
            config_path (str, optional): _description_. Defaults to None.

        Returns:
            Config: _description_
        """
        if config_path is None:
            config_path = ConfigLoader.get_config_file()
        config = configparser.ConfigParser()
        config.read(config_path)

        internal_port = config[FIELD_INTERNAL][FIELD_PORT]
        host = config[FIELD_EXTERNAL][FIELD_HOST]
        api_key = config[FIELD_EXTERNAL][FIELD_API_KEY]
        timeout_connection_start = config[FIELD_TIMEOUTS][FIELD_CON_START]
        timeout_connection_restart = config[FIELD_TIMEOUTS][FIELD_CON_RESTART]
        return Config(
            api_key=api_key,
            internal_port=internal_port,
            host=host,
            timeout_connection_start=timeout_connection_start,
            timeout_connection_restart=timeout_connection_restart
        )