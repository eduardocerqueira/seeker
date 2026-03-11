#date: 2026-03-11T17:32:37Z
#url: https://api.github.com/gists/ff4f91fd2487b9d23843e8c09adeb659
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # storage/file_based_storage.py
      2: import json
      3: import yaml
      4: import pickle
      5: from pathlib import Path
      6: from typing import Any, Dict, List
      7: from dataclasses import dataclass, asdict
      8: 
      9: @dataclass
     10: class AgentState:
     11:     """État d'un agent stocké dans des fichiers"""
     12:     agent_id: str
     13:     current_task: Dict[str, Any]
     14:     context: Dict[str, Any]
     15:     memory: List[Dict[str, Any]]
     16:     performance: Dict[str, float]
     17:     
     18:     def save(self, base_path: Path):
     19:         """Sauvegarde l'état dans des fichiers"""
     20:         agent_dir = base_path / self.agent_id
     21:         agent_dir.mkdir(exist_ok=True)
     22:         
     23:         # Sauvegarde en JSON (lisible)
     24:         with open(agent_dir / "state.json", "w") as f:
     25:             json.dump(asdict(self), f, indent=2)
     26:         
     27:         # Sauvegarde en pickle pour la performance
     28:         with open(agent_dir / "state.pkl", "wb") as f:
     29:             pickle.dump(self, f)
     30:         
     31:         # Sauvegarde du contexte séparément
     32:         context_file = agent_dir / "context" / f"context_{datetime.now().timestamp()}.json"
     33:         context_file.parent.mkdir(exist_ok=True)
     34:         with open(context_file, "w") as f:
     35:             json.dump(self.context, f, indent=2)
     36:     
     37:     @classmethod
     38:     def load(cls, agent_id: str, base_path: Path):
     39:         """Charge l'état depuis les fichiers"""
     40:         agent_dir = base_path / agent_id
     41:         
     42:         try:
     43:             # Essaye d'abord le pickle (plus rapide)
     44:             with open(agent_dir / "state.pkl", "rb") as f:
     45:                 return pickle.load(f)
     46:         except:
     47:             # Fallback sur JSON
     48:             with open(agent_dir / "state.json", "r") as f:
     49:                 data = json.load(f)
     50:                 return cls(**data)
     51: 
     52: class ProjectFileSystem:
     53:     """Gestionnaire du système de fichiers du projet"""
     54:     
     55:     def __init__(self, project_root: Path):
     56:         self.root = project_root
     57:         
     58:         # Structure de base
     59:         self.directories = {
     60:             "agents": self.root / "agents",
     61:             "context": self.root / "context",
     62:             "memory": self.root / "memory",
     63:             "workspace": self.root / "workspace",
     64:             "sprints": self.root / "sprints",
     65:             "reports": self.root / "reports",
     66:             "logs": self.root / "logs",
     67:         }
     68:         
     69:         # Initialisation
     70:         self._init_structure()
     71:     
     72:     def _init_structure(self):
     73:         """Crée la structure de dossiers"""
     74:         for dir_path in self.directories.values():
     75:             dir_path.mkdir(exist_ok=True, parents=True)
     76:             
     77:             # Sous-structure spécifique
     78:             if dir_path.name == "agents":
     79:                 for agent_type in ["coder", "tester", "architect", "documenter"]:
     80:                     (dir_path / agent_type).mkdir(exist_ok=True)
     81:             
     82:             if dir_path.name == "memory":
     83:                 for sub in ["vector_store", "history", "learnings"]:
     84:                     (dir_path / sub).mkdir(exist_ok=True)
     85:     
     86:     def get_agent_state_path(self, agent_id: str) -> Path:
     87:         """Retourne le chemin de l'état d'un agent"""
     88:         return self.directories["agents"] / agent_id
     89:     
     90:     def save_task_result(self, task_id: str, result: Dict[str, Any]):
     91:         """Sauvegarde le résultat d'une tâche"""
     92:         task_file = self.directories["sprints"] / "current" / f"task_{task_id}.json"
     93:         task_file.parent.mkdir(exist_ok=True)
     94:         
     95:         with open(task_file, "w") as f:
     96:             json.dump({
     97:                 "task_id": task_id,
     98:                 "result": result,
     99:                 "timestamp": datetime.now().isoformat(),
    100:                 "agent": result.get("agent", "unknown")
    101:             }, f, indent=2)
    102:     
    103:     def load_agent_context(self, agent_id: str, limit: int = 10) -> List[Dict]:
    104:         """Charge le contexte récent d'un agent"""
    105:         context_dir = self.get_agent_state_path(agent_id) / "context"
    106:         
    107:         if not context_dir.exists():
    108:             return []
    109:         
    110:         # Récupère les fichiers de contexte les plus récents
    111:         context_files = sorted(context_dir.glob("*.json"), 
    112:                              key=lambda x: x.stat().st_mtime, 
    113:                              reverse=True)[:limit]
    114:         
    115:         contexts = []
    116:         for file in context_files:
    117:             with open(file, "r") as f:
    118:                 contexts.append(json.load(f))
    119:         
    120:         return contexts
    121: 
    122: # Utilisation
    123: project_fs = ProjectFileSystem(Path("./project_data"))
    124: 
    125: # Sauvegarde d'état
    126: agent_state = AgentState(
    127:     agent_id="coder_001",
    128:     current_task={"id": "TASK-123", "description": "Implement login"},
    129:     context={"files": ["auth.py"], "dependencies": ["flask"]},
    130:     memory=[{"task": "previous", "result": "success"}],
    131:     performance={"accuracy": 0.95, "speed": 120}
    132: )
    133: 
    134: agent_state.save(project_fs.get_agent_state_path("coder_001"))