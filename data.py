import aiohttp
import asyncio
import json
import re

from bs4 import BeautifulSoup


async def get_shops_addresses(session):
    async with session.get('https://perekrestok-promo.ru/store/g-moskva') as response:
        html = await response.text()
    tree = BeautifulSoup(html, 'html.parser')
    feature = 'Магазин Перекресток по адресу '
    raw_links = tree.body.find_all('a', string=re.compile(feature))
    links = [(link.text.replace(feature, '')) for link in raw_links]
    return links


async def coordinate_from_address(session, address):
    params = [
        ('key', 'AIzaSyBv5ugTR5L_tNtHQPKFDvHJV_-oxnuM2N8'),
        ('address', address)
    ]
    async with session.get('https://maps.googleapis.com/maps/api/geocode/json', params=params) as response:
        text = await response.text()
        if response.status == 200:
            data = json.loads(text)
            try:
                position = data['results'][0]['geometry']['location']
                return (float(position['lat']), float(position['lng']))
            except (KeyError, IndexError):
                pass
    return None


async def load_points():
    chunk_size = 30
    session = aiohttp.ClientSession()
    addresses = await get_shops_addresses(session)
    tasks = [coordinate_from_address(session, address) for address in addresses]
    results = []
    while len(tasks):
        chunk = tasks[:chunk_size]
        tasks = tasks[chunk_size:]
        finished, unfinished = await asyncio.wait(chunk)
        results.extend([task.result() for task in finished if task.result() is not None])
        await asyncio.sleep(1)
    await session.close()
    return results
