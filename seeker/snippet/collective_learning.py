#date: 2026-03-16T17:43:40Z
#url: https://api.github.com/gists/befd3d7c917b9b1c1c82a6930ce45c5a
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # learning/collective_learning.py
      2: import hashlib
      3: from typing import Dict, List
      4: from pathlib import Path
      5: 
      6: class CollectiveLearningSystem:
      7:     """Système d'apprentissage partagé basé sur fichiers"""
      8:     
      9:     def __init__(self, knowledge_base_path: Path):
     10:         self.knowledge_base = knowledge_base_path
     11:         self.knowledge_base.mkdir(exist_ok=True)
     12:         
     13:         # Sous-répertoires
     14:         self.patterns_dir = self.knowledge_base / "patterns"
     15:         self.solutions_dir = self.knowledge_base / "solutions"
     16:         self.mistakes_dir = self.knowledge_base / "mistakes"
     17:         self.best_practices_dir = self.knowledge_base / "best_practices"
     18:         
     19:         for dir in [self.patterns_dir, self.solutions_dir, 
     20:                    self.mistakes_dir, self.best_practices_dir]:
     21:             dir.mkdir(exist_ok=True)
     22:     
     23:     def store_solution(self, problem_hash: str, solution: Dict, agent_id: str):
     24:         """Stocke une solution à un problème"""
     25:         solution_file = self.solutions_dir / f"{problem_hash}.json"
     26:         
     27:         data = {
     28:             "problem_hash": problem_hash,
     29:             "solution": solution,
     30:             "agent": agent_id,
     31:             "timestamp": datetime.now().isoformat(),
     32:             "effectiveness": 1.0,  # Sera mis à jour avec le feedback
     33:             "usage_count": 1
     34:         }
     35:         
     36:         if solution_file.exists():
     37:             # Mise à jour du compteur d'utilisation
     38:             with open(solution_file, "r") as f:
     39:                 existing = json.load(f)
     40:                 existing["usage_count"] += 1
     41:             
     42:             data = existing
     43:         
     44:         with open(solution_file, "w") as f:
     45:             json.dump(data, f, indent=2)
     46:     
     47:     def find_similar_solutions(self, problem_description: str, 
     48:                               threshold: float = 0.8) -> List[Dict]:
     49:         """Trouve des solutions similaires"""
     50:         problem_hash = self._hash_description(problem_description)
     51:         
     52:         solutions = []
     53:         for solution_file in self.solutions_dir.glob("*.json"):
     54:             with open(solution_file, "r") as f:
     55:                 solution_data = json.load(f)
     56:                 
     57:                 # Calcul de similarité basique
     58:                 similarity = self._calculate_similarity(
     59:                     problem_hash, 
     60:                     solution_data["problem_hash"]
     61:                 )
     62:                 
     63:                 if similarity >= threshold:
     64:                     solutions.append(solution_data)
     65:         
     66:         return sorted(solutions, key=lambda x: x["effectiveness"], reverse=True)
     67:     
     68:     def learn_from_mistake(self, agent_id: str, task: Dict, 
     69:                           error: str, correction: Dict):
     70:         """Apprend d'une erreur et la partage"""
     71:         mistake_id = f"{agent_id}_{datetime.now().timestamp()}"
     72:         mistake_file = self.mistakes_dir / f"{mistake_id}.json"
     73:         
     74:         with open(mistake_file, "w") as f:
     75:             json.dump({
     76:                 "agent": agent_id,
     77:                 "task": task,
     78:                 "error": error,
     79:                 "correction": correction,
     80:                 "timestamp": datetime.now().isoformat(),
     81:                 "learned_by": [agent_id]  # Liste des agents ayant appris
     82:             }, f, indent=2)
     83:         
     84:         # Notifie les autres agents
     85:         self._broadcast_learning(agent_id, "mistake_learned", mistake_id)
     86:     
     87:     def _broadcast_learning(self, from_agent: str, learning_type: str, 
     88:                           content_id: str):
     89:         """Diffuse un apprentissage à tous les agents"""
     90:         learning_event = {
     91:             "type": learning_type,
     92:             "from": from_agent,
     93:             "content_id": content_id,
     94:             "timestamp": datetime.now().isoformat()
     95:         }
     96:         
     97:         # Écrit dans un fichier partagé que tous les agents lisent
     98:         event_file = self.knowledge_base / "learning_events.jsonl"
     99:         
    100:         with open(event_file, "a") as f:
    101:             f.write(json.dumps(learning_event) + "\n")
    102:     
    103:     @staticmethod
    104:     def _hash_description(description: str) -> str:
    105:         """Hash une description pour comparaison"""
    106:         return hashlib.md5(description.encode()).hexdigest()[:10]
    107:     
    108:     @staticmethod
    109:     def _calculate_similarity(hash1: str, hash2: str) -> float:
    110:         """Calcule la similarité entre deux hashes"""
    111:         # Simple similarité de Levenshtein normalisée
    112:         from Levenshtein import distance
    113:         max_len = max(len(hash1), len(hash2))
    114:         if max_len == 0:
    115:             return 1.0
    116:         return 1 - distance(hash1, hash2) / max_len