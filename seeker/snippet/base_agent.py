#date: 2026-03-11T17:32:40Z
#url: https://api.github.com/gists/af28814cb19d96d15e7cb763f76c17cf
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: """
      2: Classe de base pour tous les agents - Version corrigée
      3: """
      4: from abc import ABC, abstractmethod
      5: from typing import Dict, Any
      6: import logging
      7: 
      8: class BaseAgent(ABC):
      9:     """Classe abstraite de base pour tous les agents"""
     10:     
     11:     def __init__(self, config_path: str = ""):
     12:         self.config_path = config_path
     13:         self.logger = logging.getLogger(self.__class__.__name__)
     14:         self.agent_id = f"{self.__class__.__name__.lower()}_01"
     15:         
     16:         self.logger.info(f"Agent {self.agent_id} initialisé")
     17:     
     18:     @abstractmethod
     19:     async def execute(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
     20:         """Méthode abstraite pour exécuter une tâche"""
     21:         pass
     22:     
     23:     async def health_check(self) -> Dict[str, Any]:
     24:         """Vérifie la santé de l'agent"""
     25:         return {
     26:             "agent_id": self.agent_id,
     27:             "status": "healthy",
     28:             "type": self.__class__.__name__
     29:         }