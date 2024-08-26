***********
Environment
***********

This environment corresponds to the game PenQuest. In case you are not
familiar with PenQuest please first read the :doc:`recap <recap>`.

+-----------------------+-----------------------------------------------+
| Action Space          | `Sequence(Discrete(0, 1e10))`                 |
+-----------------------+-----------------------------------------------+
| Observation Space     |  `Dict(...)`                                  |
+-----------------------+-----------------------------------------------+
|| import               || `import penquest-env`                        |
||                      || `penquest-env.start()`                       |
||                      || `gymnasium.make('penquest-env/PenQuest-v0')` |
+-----------------------+-----------------------------------------------+



.. autoclass:: penquest_env.PenQuestEnv.PenQuestEnv