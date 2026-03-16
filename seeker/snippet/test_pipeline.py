#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import asyncio
      2: from agents.tester.tester import TesterAgent
      3: from agents.formal_verification.formal_verification import FormalVerificationAgent
      4: 
      5: async def test_pipeline():
      6:     print('🧪 PIPELINE COMPLET DE VÉRIFICATION')
      7:     print('='*50)
      8:     
      9:     # 1. Agent de test
     10:     tester = TesterAgent()
     11:     await tester.initialize()
     12:     print(f'✅ TesterAgent: {tester.status.value}')
     13:     
     14:     # 2. Agent de vérification formelle
     15:     formal = FormalVerificationAgent()
     16:     await formal.initialize()
     17:     print(f'✅ FormalAgent: {formal.status.value}')
     18:     
     19:     # 3. Génération de tests
     20:     test_result = await tester._generate_tests(
     21: "**********": 'Token', 'framework': 'foundry'},
     22:         {}
     23:     )
     24:     print(f'✅ Tests générés: {test_result["generated_file"]}')
     25:     
     26:     # 4. Vérification formelle
     27: "**********"
     28:     print(f'✅ Preuve générée: {proof.id}')
     29:     print(f'✅ Propriétés vérifiées: {len(proof.verified_properties)}')
     30:     
     31:     print('='*50)
     32:     print('🎉 PIPELINE COMPLET OPÉRATIONNEL')
     33:     print('='*50)
     34: 
     35: asyncio.run(test_pipeline())    print('='*50)
     34: 
     35: asyncio.run(test_pipeline())