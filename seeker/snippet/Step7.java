//date: 2022-03-18T17:03:42Z
//url: https://api.github.com/gists/a6861fd56127fb8e5a1d622be8807053
//owner: https://api.github.com/users/davimh22

for(int i = 0; i < animals.size(); i++)
		{
			Animal a = animals.get(i);
			System.out.println(a.getName() + "\n========================");
			a.Entertainment();
			System.out.println();
			System.out.println("Hunger: " + a.getHunger() + "\nIntimidation: " + a.getIntima() + "\nLikes People: " + a.getPeople());
			if(a.equals(p)|| a.equals(pi))
			{
				if(p.getFedora()|| pi.getFedora())
				{
					System.out.println("Perry's Got his fedora and he's hungry for JUSTICE!!!");
				}
				
			}
			if(a.equals(d))
			{
				if(d.getJersey())
				{
					System.out.println("Draymond's got his jersey and is ready to pick up some rebounds and technical fouls.");
				}
			}
			if(a.equals(s))
			{
				if(s.getHair())
				{
					System.out.println("Young Thug's got that pink hair now for his slimes");
				}
			}
			if(a.equals(g))
			{
				if(g.getGlasses())
				{
					System.out.println("Future's feeling SENSATIONAL with his glasses on and his cheesecake");
				}
			}
			System.out.println();
			System.out.println();
		}