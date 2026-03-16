#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: """
      2: Learning Agent Package - Intelligence Artificielle pour l'optimisation du pipeline
      3: """
      4: 
      5: # Import depuis le module principal
      6: from .agent import (
      7:     LearningAgent,
      8:     LearningTaskType,
      9:     ModelType,
     10:     ConfidenceLevel,
     11:     TrainingExample,
     12:     ModelMetadata,
     13:     Insight,
     14:     create_learning_agent
     15: )
     16: 
     17: __all__ = [
     18:     'LearningAgent',
     19:     'LearningTaskType',
     20:     'ModelType',
     21:     'ConfidenceLevel',
     22:     'TrainingExample',
     23:     'ModelMetadata',
     24:     'Insight',
     25:     'create_learning_agent'
     26: ]
     27: 
     28: __version__ = '2.0.0'