import asyncio

from flask import Flask, send_file
from aiohttp import web

from tile import gen_tile
from data import load_points


loop = asyncio.get_event_loop()
app = Flask(__name__)
points = []


@app.route("/")
def home():
    return "hey!"


@app.route('/tiles/<int:zoom>/<int:x>/<int:y>.png')
def tiles(zoom, x, y):
    return send_file(gen_tile(zoom, x, y, points))


if __name__ == "__main__":
    points = loop.run_until_complete(load_points())
    app.run(host='0.0.0.0', port=8888, debug=True, use_reloader=True)
