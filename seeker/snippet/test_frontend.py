#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import asyncio
      2: from agents.frontend_web3.frontend_agent import FrontendWeb3Agent
      3: 
      4: async def test_frontend():
      5:     print("🎨 TEST AGENT FRONTEND WEB3")
      6:     print("="*50)
      7:     
      8:     agent = FrontendWeb3Agent()
      9:     await agent.initialize()
     10:     
     11:     # Générer un projet Next.js
     12:     project = await agent.generate_project(
     13:         project_name="CryptoKitties Clone",
     14: "**********"
     15:         components=["mint_page", "nft_gallery"],
     16:         framework="nextjs"
     17:     )
     18:     
     19:     print(f"\n📦 Projet généré!")
     20:     print(f"  📁 Output: {project.output_path}")
     21:     print(f"  🖥️  Framework: {project.framework.value}")
     22:     print(f"  📄 Composants: {len(project.components)}")
     23:     print(f"\n✅ Pour lancer le projet:")
     24:     print(f"  cd {project.output_path}")
     25:     print(f"  npm install")
     26:     print(f"  npm run dev")
     27:     print(f"\n🌐 http://localhost:3000")
     28: 
     29: asyncio.run(test_frontend())     29: asyncio.run(test_frontend())