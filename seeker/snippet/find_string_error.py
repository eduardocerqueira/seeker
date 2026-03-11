#date: 2026-03-11T17:32:37Z
#url: https://api.github.com/gists/ff4f91fd2487b9d23843e8c09adeb659
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # find_string_error.py
      2: import re
      3: 
      4: def find_unterminated_strings(filepath):
      5:     """Trouve les chaînes non terminées dans un fichier Python"""
      6:     print(f"🔍 Recherche d'erreurs dans: {filepath}")
      7:     
      8:     with open(filepath, 'r', encoding='utf-8') as f:
      9:         content = f.read()
     10:     
     11:     # Chercher les docstrings non fermées
     12:     lines = content.split('\n')
     13:     
     14:     for i, line in enumerate(lines, 1):
     15:         # Compter les guillemets
     16:         triple_double = line.count('"""')
     17:         triple_single = line.count("'''")
     18:         
     19:         # Si nombre impair de guillemets triples
     20:         if triple_double % 2 != 0 or triple_single % 2 != 0:
     21:             print(f"⚠️  Ligne {i}: Chaîne triple potentiellement non fermée")
     22:             print(f"   Contenu: {line[:100]}...")
     23:         
     24:         # Chercher les guillemets simples/doubles non fermés
     25:         single_quotes = re.findall(r"(?<!\')'(?!\')", line)
     26:         double_quotes = re.findall(r'(?<!")\"(?!")', line)
     27:         
     28:         if len(single_quotes) % 2 != 0:
     29:             print(f"⚠️  Ligne {i}: Guillemet simple non fermé")
     30:             print(f"   Contenu: {line[:100]}...")
     31:         
     32:         if len(double_quotes) % 2 != 0:
     33:             print(f"⚠️  Ligne {i}: Guillemet double non fermé")
     34:             print(f"   Contenu: {line[:100]}...")
     35:     
     36:     # Vérifier autour de la ligne 1940 spécifiquement
     37:     print(f"\n🔍 Vérification détaillée autour de la ligne 1940:")
     38:     start = max(1930, 0)
     39:     end = min(1950, len(lines))
     40:     
     41:     for i in range(start, end):
     42:         print(f"{i:4}: {lines[i]}")
     43:     
     44:     print("\n💡 CONSEIL: Vérifiez les docstrings autour de cette ligne")
     45: 
     46: if __name__ == "__main__":
     47:     find_unterminated_strings("agents/coder/coder.py")