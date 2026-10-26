"""Procedural aerial imagery for the product-concept demo.

NOT training data and NOT model input. This renders what a 256m tile of a
small settlement looks like from above, so the concept console can show the
interface against something that reads as imagery instead of coloured blocks.
Everything here is drawn, not photographed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

RNG = np.random.default_rng


def value_noise(h, w, scale, rng, octaves=4):
    """Smooth multi-octave noise in [0,1]."""
    out = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        s = max(2, int(scale / (2 ** o)))
        g = rng.random((s, s)).astype(np.float32)
        z = ndimage.zoom(g, (h / s, w / s), order=3)[:h, :w]
        out += z * amp
        total += amp
        amp *= 0.5
    out /= total
    return np.clip(out, 0, 1)


def tint(noise, lo, hi):
    """Blend between two RGB colours by a noise field in [0,1]."""
    lo = np.array(lo, dtype=np.float32)
    hi = np.array(hi, dtype=np.float32)
    n = noise[..., None].astype(np.float32)
    return lo * (1 - n) + hi * n


class Tile:
    def __init__(self, size=512, seed=0):
        self.n = size
        self.rng = RNG(seed)
        self.buildings = []

    # ---------------------------------------------------------------- ground
    def ground(self):
        n = self.n
        rng = self.rng
        soil = value_noise(n, n, 48, rng)
        veg = value_noise(n, n, 22, rng)
        img = tint(soil, (118, 116, 86), (142, 138, 104))
        grass = tint(veg, (96, 118, 68), (126, 146, 86))
        m = (veg > 0.52)[..., None]
        img = np.where(m, grass, img)
        # scattered canopy
        tree = value_noise(n, n, 10, rng, octaves=3)
        canopy = (tree > 0.70) & (veg > 0.45)
        shade = ndimage.shift(canopy.astype(np.float32), (4, 4), order=0)
        img = img * (1 - 0.22 * shade[..., None])
        img = np.where(canopy[..., None],
                       tint(tree, (58, 84, 48), (84, 112, 62)), img)
        self.img = img.astype(np.float32)

    # ----------------------------------------------------------------- water
    def river(self):
        n = self.n
        rng = self.rng
        yy, xx = np.mgrid[0:n, 0:n]
        wob = value_noise(n, n, 16, rng) * 46 - 23
        centre = 0.62 * n + (xx - n / 2) * 0.42 + wob
        d = np.abs(yy - centre)
        width = 20 + value_noise(n, n, 20, rng) * 10
        water = d < width
        shallow = (d >= width) & (d < width + 6)
        ripple = value_noise(n, n, 8, rng)
        self.img = np.where(water[..., None],
                            tint(ripple, (48, 78, 96), (74, 110, 128)), self.img)
        self.img = np.where(shallow[..., None],
                            np.array([120, 126, 108], dtype=np.float32), self.img)
        self.water = water

    # ------------------------------------------------------------------ road
    def roads(self):
        n = self.n
        rng = self.rng
        road = np.zeros((n, n), dtype=bool)
        for frac in (0.34, 0.72):
            c = int(n * frac) + int(rng.integers(-10, 10))
            road[c - 7:c + 7, :] = True
        c = int(n * 0.5) + int(rng.integers(-12, 12))
        road[:, c - 7:c + 7] = True
        grain = value_noise(n, n, 30, rng)
        self.img = np.where(road[..., None],
                            tint(grain, (104, 102, 100), (132, 130, 126)), self.img)
        self.road = road

    # ------------------------------------------------------------- buildings
    def place(self, kinds, count=17):
        n = self.n
        rng = self.rng
        roofs = {
            'house': [(150, 108, 88), (176, 132, 104)],
            'apartments': [(126, 118, 122), (156, 148, 150)],
            'warehouse': [(132, 136, 142), (166, 170, 176)],
            'shop': [(158, 132, 86), (186, 160, 108)],
            'clinic': [(196, 198, 196), (222, 224, 222)],
            'market': [(160, 96, 78), (190, 124, 100)],
            'bridge': [(140, 128, 110), (168, 156, 136)],
        }
        placed = []
        tries = 0
        while len(placed) < count and tries < 900:
            tries += 1
            kind = kinds[len(placed) % len(kinds)]
            w = int(rng.integers(30, 58))
            h = int(rng.integers(26, 52))
            x = int(rng.integers(6, n - w - 6))
            y = int(rng.integers(6, n - h - 6))
            box = (x, y, x + w, y + h)
            if self.road[y:y + h, x:x + w].any() or self.water[y:y + h, x:x + w].any():
                continue
            if any(not (box[2] < b[0] - 8 or box[0] > b[2] + 8 or
                        box[3] < b[1] - 8 or box[1] > b[3] + 8) for b, _k in placed):
                continue
            placed.append((box, kind))
        self.buildings = placed

        shadow = Image.new('L', (n, n), 0)
        sd = ImageDraw.Draw(shadow)
        for (x0, y0, x1, y1), _k in placed:
            sd.rectangle([x0 + 7, y0 + 9, x1 + 7, y1 + 9], fill=150)
        shadow = np.array(shadow.filter(ImageFilter.GaussianBlur(4)), dtype=np.float32) / 255.0
        self.img = self.img * (1 - 0.45 * shadow[..., None])

        for (x0, y0, x1, y1), kind in placed:
            lo, hi = roofs[kind]
            bh, bw = y1 - y0, x1 - x0
            grain = value_noise(bh, bw, 12, rng)
            roof = tint(grain, lo, hi)
            # ridge line down the long axis
            if bw >= bh:
                roof[bh // 2 - 1:bh // 2 + 1, :] *= 0.86
            else:
                roof[:, bw // 2 - 1:bw // 2 + 1] *= 0.86
            roof[0:2, :] *= 1.10
            roof[:, 0:2] *= 1.06
            self.img[y0:y1, x0:x1] = roof

    # ---------------------------------------------------------------- damage
    def damaged(self, levels):
        """levels: dict index -> 0..3. Returns a new post-event image."""
        rng = self.rng
        post = self.img.copy()
        n = self.n
        for i, ((x0, y0, x1, y1), _kind) in enumerate(self.buildings):
            lv = levels.get(i, 0)
            if lv == 0:
                continue
            bh, bw = y1 - y0, x1 - x0
            frac = {1: 0.22, 2: 0.58, 3: 0.92}[lv]
            dark = {1: 0.88, 2: 0.62, 3: 0.40}[lv]
            patch = post[y0:y1, x0:x1] * dark
            rub = value_noise(bh, bw, 6, rng)
            hit = rng.random((bh, bw)) < frac
            rubble = tint(rub, (92, 84, 76), (132, 122, 110))
            patch = np.where(hit[..., None], rubble, patch)
            post[y0:y1, x0:x1] = patch
            if lv == 3:
                # debris thrown outward
                for _ in range(int(bw * bh * 0.012)):
                    dx = int(rng.normal(0, bw * 0.55))
                    dy = int(rng.normal(0, bh * 0.55))
                    px = np.clip(x0 + bw // 2 + dx, 0, n - 3)
                    py = np.clip(y0 + bh // 2 + dy, 0, n - 3)
                    post[py:py + 2, px:px + 2] *= 0.70
        return post


def finish(arr):
    a = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(a)
    return im.filter(ImageFilter.GaussianBlur(0.4))


def make_pair(seed, levels, size=512):
    kinds = ['house', 'apartments', 'warehouse', 'shop', 'clinic', 'market', 'house']
    t = Tile(size, seed)
    t.ground()
    t.river()
    t.roads()
    t.place(kinds, count=17)
    pre = finish(t.img)
    post = finish(t.damaged(levels))
    return pre, post, t.buildings


if __name__ == '__main__':
    rng = RNG(7)
    levels = {i: int(rng.integers(0, 4)) for i in range(17)}
    pre, post, b = make_pair(3, levels)
    pre.save('/home/claude/vision/pre.png')
    post.save('/home/claude/vision/post.png')
    print(f'{len(b)} buildings; damage levels {sorted(levels.values())}')
