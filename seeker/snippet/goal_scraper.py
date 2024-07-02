#date: 2024-07-02T16:52:06Z
#url: https://api.github.com/gists/919bb2df4453bae095f1ca8518fbac72
#owner: https://api.github.com/users/cleitonleonel

import json
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from datetime import datetime


class GoalScraper:
    def __init__(self, base_url, edition, country, category, league, league_id):
        self.base_url = base_url
        self.edition = edition
        self.country = country
        self.category = category
        self.league = league
        self.league_id = league_id

    async def fetch(self, session, url):
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()

    async def fetch_json(self, session, url, params=None):
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def get_props(self):
        async with aiohttp.ClientSession() as session:
            html = await self.fetch(session,
                                    f'{self.base_url}/{self.edition}/{self.league}/{self.category}/{self.league_id}')
            soup = BeautifulSoup(html, 'html.parser')
            script = soup.find('script', {"id": "__NEXT_DATA__"})
            if not script:
                raise ValueError("Não foi possível encontrar o script com dados da página.")
            json_data = json.loads(script.text).get('props')
            return json_data

    async def get_matches(self, data):
        matches = data['gamesets'][0]['matches']
        round_name = data['gamesets'][0]['name']
        print(f'Rodada: {round_name}')
        future_matches_count = 0
        today = datetime.today()

        for match in matches:
            team_a = match['teamA']['name']
            team_b = match['teamB']['name']
            match_date_str = match['startDate']
            match_date = datetime.strptime(match_date_str, '%Y-%m-%dT%H:%M:%S.%fZ')

            if match_date >= today:
                future_matches_count += 1

            formatted_date = match_date.strftime('%d/%m/%Y')
            score_team_a = match.get('score', {}).get('teamA', '') if match.get('score') else ''
            score_team_b = match.get('score', {}).get('teamB', '') if match.get('score') else ''

            print(f'Data: {formatted_date} - {team_a} {score_team_a} X {score_team_b} {team_b}')

        if future_matches_count > 0:
            print('Rodadas Futuras!!')
        print(f'Fim da rodada: {round_name}\n{"=" * 50}\n')

    async def get_games(self):
        try:
            props = await self.get_props()
            game_sets = props['pageProps']['content']['gamesets']

            async with aiohttp.ClientSession() as session:
                tasks = []
                for game_set in game_sets:
                    payload = {
                        "edition": self.edition,
                        "id": self.league_id,
                        "country": self.country,
                        "gameSetTypeIds": game_set.get("gameSetTypeId")
                    }
                    tasks.append(self.fetch_json(session, f'{self.base_url}/api/competition-matches', params=payload))

                results = await asyncio.gather(*tasks)

                for result_data in results:
                    await self.get_matches(result_data)

        except aiohttp.ClientError as e:
            print(f"Erro ao fazer requisição: {e}")
        except ValueError as e:
            print(f"Erro de valor: {e}")
        except KeyError as e:
            print(f"Chave não encontrada: {e}")


if __name__ == "__main__":
    scraper = GoalScraper(
        base_url='https://www.goal.com',
        edition='br',
        country='BR',
        category='partidas-resultados',
        league='brasileirao',
        league_id='scf9p4y91yjvqvg5jndxzhxj'
    )
    asyncio.run(scraper.get_games())
