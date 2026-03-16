#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import asyncio
      2: from agents.fuzzing_simulation.fuzzing_agent import FuzzingSimulationAgent
      3: 
      4: async def test_fuzzing():
      5:     print("🧪 TEST AGENT DE FUZZING")
      6:     print("="*50)
      7:     
      8:     agent = FuzzingSimulationAgent()
      9:     await agent.initialize()
     10:     
     11:     # Lancer une campagne
     12:     campaign = await agent.run_fuzzing_campaign(
     13: "**********"
     14:         campaign_name="Test Fuzzing",
     15:         template="comprehensive"
     16:     )
     17:     
     18:     # Afficher les résultats
     19:     print(f"\n📊 Résultats:")
     20:     print(f"  ✅ Campagne: {campaign.id}")
     21:     print(f"  ✅ Statut: {campaign.status}")
     22:     print(f"  ✅ Tests: {campaign.total_tests}")
     23:     print(f"  🔴 Vulnérabilités: {len(campaign.vulnerabilities)}")
     24:     
     25:     for vuln in campaign.vulnerabilities[:3]:
     26:         print(f"\n  🔥 {vuln['severity'].upper()}: {vuln['description']}")
     27:         print(f"     → {vuln['remediation']}")
     28:     
     29:     print(f"\n  📄 Rapport: {campaign.report_path}")
     30: 
     31: asyncio.run(test_fuzzing())   31: asyncio.run(test_fuzzing())