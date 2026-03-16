#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import asyncio
      2: from agents.frontend_web3.frontend_agent import FrontendWeb3Agent
      3: 
      4: async def test_frontend_config():
      5:     print("🎨 TEST AGENT AVEC CONFIGURATION YAML")
      6:     print("="*50)
      7:     
      8:     # Utiliser le fichier de config
      9:     agent = FrontendWeb3Agent("agents/frontend_web3/config.yaml")
     10:     await agent.initialize()
     11:     
     12:     print(f"✅ Agent: {agent._display_name}")
     13:     print(f"✅ Version: {agent._version}")
     14:     print(f"✅ Capacités: {len(agent._agent_config['agent']['capabilities'])}")
     15:     
     16:     # Extraire l'ABI du nouveau contrat
     17:     abi_info = await agent.extract_contract_abi("./contracts/SimpleNFT.sol")
     18:     
     19:     if abi_info["abi"]:
     20:         print(f"✅ ABI extraite: {len(abi_info['abi'])} fonctions")
     21:     else:
     22:         print("⚠️ ABI non trouvée - compilation nécessaire")
     23:     
     24:     # Générer le projet
     25:     project = await agent.generate_project(
     26:         project_name="NFTCollection",
     27:         contract_paths=["./contracts/SimpleNFT.sol"],
     28:         components=["mint_page", "nft_gallery", "dashboard"],
     29:         framework="nextjs"
     30:     )
     31:     
     32:     print(f"\n📦 Projet généré!")
     33:     print(f"  📁 Output: {project.output_path}")
     34:     print(f"  📄 Composants: {len(project.components)}")
     35:     print(f"  📄 Contrats: {len(project.contracts)}")
     36:     print(f"\n🚀 Lancer le projet:")
     37:     print(f"  cd {project.output_path}")
     38:     print(f"  npm install")
     39:     print(f"  npm run dev")
     40: 
     41: asyncio.run(test_frontend_config())