#date: 2026-03-11T17:32:40Z
#url: https://api.github.com/gists/af28814cb19d96d15e7cb763f76c17cf
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import asyncio
      2: from agents.frontend_web3.frontend_agent import __version__
      3: from agents.frontend_web3.frontend_agent import FrontendWeb3Agent
      4: 
      5: 
      6: async def test_frontend_2_0():
      7:     print("🎨 TEST FRONTEND WEB3 2.0 - BANKING + LENDING + NFT")
      8:     print("="*60)
      9:     
     10:     agent = FrontendWeb3Agent("agents/frontend_web3/config.yaml")
     11:     await agent.initialize()
     12:     
     13:     print(f"✅ Agent version: {__version__}")
     14:     print(f"✅ Capacités 2.0: {len(agent._load_capabilities_2_0())}")
     15:     print(f"✅ Templates 2.0: {list(agent._templates_2_0.keys())}")
     16:     
     17:     # Générer un projet avec les composants 2.0
     18:     project = await agent.generate_project(
     19:         project_name="BankingWeb3",
     20:         contract_paths=["./contracts/SimpleNFT.sol"],
     21:         components=[
     22:             "banking_dashboard_2_0",
     23:             "virtual_cards",
     24:             "savings_pods",
     25:             "credit_scoring",
     26:             "nft_lending",
     27:             "defi_composer"
     28:         ],
     29:         framework="nextjs"
     30:     )
     31:     
     32:     print(f"\n📦 Projet 2.0 généré!")
     33:     print(f"  📁 Output: {project.output_path}")
     34:     print(f"  📄 Composants 2.0: {len(project.components)}")
     35:     print(f"\n🚀 Lancer le projet:")
     36:     print(f"  cd {project.output_path}")
     37:     print(f"  npm install")
     38:     print(f"  npm run dev")
     39:     print(f"\n🌐 http://localhost:3000/banking")
     40: 
     41: asyncio.run(test_frontend_2_0())