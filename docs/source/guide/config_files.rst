Configuration Files
===================

Configuration files store all the configuration data that is used for the 
PenQuest-Env package. This includes API-keys, host adresses, ports, etc. Such
files usually have the file ending `.ini` and their file path should be provided
when calling the :ref:`penquest_env.start() <penquest_env.start>` function. 

You can use the default configuration of `default_config.ini` from within the 
package as a starting point to create your own
`config.ini` file:

.. code-block::

    [internal]
    port = 50000

    [external]
    host = localhost
    port = 5000
    api_key = 

    [timeouts]
    connection_start = 300
    connection_restart = 1

PenQuest-Env configuration files are split in multiple sections. These currently
are:

* internal
    contains all fields in relation to the internal communication 
    between the websocket process and multiple environment processes
* external
    contains all fields in relation to the websocket communication
* timeouts 
    contains all fields in relation to how long the websocket process
    waits after certain events before shuting down
