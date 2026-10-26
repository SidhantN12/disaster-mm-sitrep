(function () {
  'use strict';
  var DATA = window.__DEMO__;
  var TILES = DATA.tiles;
  var DN = DATA.meta.damage_names;

  var state = { tile: 1, layer: 'post', word: 0, labels: true };

  var $ = function (id) { return document.getElementById(id); };

  /* ---------- tile strip ---------- */
  function buildStrip() {
    var strip = $('strip');
    TILES.forEach(function (t, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip' + (t.n_reports === 0 ? ' k0' : '');
      b.setAttribute('aria-pressed', String(i === state.tile));
      b.innerHTML = '<img src="' + t.post + '" alt=""><span>' + String(i).padStart(2, '0') + '</span>';
      b.addEventListener('click', function () { select(i); });
      strip.appendChild(b);
    });
  }

  function syncStrip() {
    var kids = $('strip').children;
    for (var i = 0; i < kids.length; i++) {
      kids[i].setAttribute('aria-pressed', String(i === state.tile));
    }
  }

  /* ---------- image + attention ---------- */
  function renderStage() {
    var t = TILES[state.tile];
    $('stageImg').src = state.layer === 'pre' ? t.pre : t.post;
    $('footTile').textContent = 'tile ' + String(state.tile).padStart(2, '0');

    var tabs = document.querySelectorAll('.imgtabs button');
    for (var i = 0; i < tabs.length; i++) {
      tabs[i].setAttribute('aria-pressed', String(tabs[i].dataset.layer === state.layer));
    }
    renderAttention();
    renderIcons();
  }


  /* ---------- structure icons (annotation overlay, not model input) ---------- */
  var GLYPH = {
    house:      'M18,52 L50,24 L82,52 M27,46 L27,80 L73,80 L73,46',
    warehouse:  'M18,80 L18,44 L50,26 L82,44 L82,80 Z M36,80 L36,58 L64,58 L64,80',
    shop:       'M24,44 L76,44 L76,80 L24,80 Z M20,44 Q33,28 46,44 Q59,28 72,44 L80,44',
    clinic:     'M50,26 L50,74 M26,50 L74,50',
    market:     'M50,30 L50,78 M22,48 Q50,18 78,48 Z',
    apartments: 'M32,80 L32,22 L68,22 L68,80 M42,34 L42,42 M58,34 L58,42 M42,54 L42,62 M58,54 L58,62',
    bridge:     'M16,64 Q50,30 84,64 M16,64 L16,76 M84,64 L84,76'
  };

  function renderIcons() {
    var layer = $('iconLayer');
    if (!layer) return;
    var t = TILES[state.tile];
    var st = (t && t.structures) || [];
    if (!st.length) { layer.innerHTML = ''; return; }
    var parts = ['<svg viewBox="0 0 100 100" preserveAspectRatio="none">'];
    for (var i = 0; i < st.length; i++) {
      var s = st[i];
      var pad = 0.26;
      var x = (s.x + s.w * pad) * 100;
      var y = (s.y + s.h * pad) * 100;
      var w = s.w * (1 - 2 * pad) * 100;
      var h = s.h * (1 - 2 * pad) * 100;
      var d = GLYPH[s.k];
      if (!d) continue;
      parts.push(
        '<svg x="' + x.toFixed(2) + '" y="' + y.toFixed(2) +
        '" width="' + w.toFixed(2) + '" height="' + h.toFixed(2) +
        '" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">' +
        '<path class="ic-h" d="' + d + '"/><path class="ic" d="' + d + '"/></svg>'
      );
    }
    parts.push('</svg>');
    layer.innerHTML = parts.join('');
    layer.className = state.labels ? 'on' : '';
  }

  function renderAttention() {
    var t = TILES[state.tile];
    var grid = $('attGrid');
    var a = t.attention[state.word];
    if (!a) {
      grid.innerHTML = '';
      $('footAtt').textContent = '';
      $('spread').innerHTML = '';
      return;
    }

    // map values are each patch's attention as a multiple of uniform (x100),
    // so 100 is exactly uniform. The spread is tiny, so stretch contrast
    // across this token's own range rather than pretending it is large.
    var lo = Infinity;
    var hi = -Infinity;
    for (var i = 0; i < a.map.length; i++) {
      if (a.map[i] < lo) lo = a.map[i];
      if (a.map[i] > hi) hi = a.map[i];
    }
    var span = Math.max(1, hi - lo);

    var html = '';
    for (var p = 0; p < a.map.length; p++) {
      var n = (a.map[p] - lo) / span;
      var op = n < 0.45 ? 0 : (n - 0.45) / 0.55 * 0.78;
      html += '<i style="opacity:' + op.toFixed(3) + '"></i>';
    }
    grid.innerHTML = html;
    $('footAtt').textContent = 'patch attention for “' + a.w + '”';
    $('spread').innerHTML =
      'This word’s attention is spread almost evenly over all 256 patches — ' +
      'strongest patch <b>' + (hi / 100).toFixed(2) + '×</b> uniform, ' +
      'weakest <b>' + (lo / 100).toFixed(2) + '×</b>. ' +
      'The shading stretches that ' + (a.spread * 100).toFixed(1) + '% range to fill the square.';
  }

  function renderWords() {
    var t = TILES[state.tile];
    var box = $('words');
    box.innerHTML = '';
    t.attention.forEach(function (a, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.textContent = a.w;
      b.setAttribute('aria-pressed', String(i === state.word));
      b.addEventListener('click', function () {
        state.word = i;
        renderWords();
        renderAttention();
        renderMass();
      });
      box.appendChild(b);
    });
  }

  function renderMass() {
    var t = TILES[state.tile];
    var a = t.attention[state.word];
    var bar = $('massbar');
    if (!a) { bar.innerHTML = ''; return; }
    var img = Math.max(0, Math.min(1, a.img));
    var rep = Math.max(0, Math.min(1, a.rep));
    var ctx = Math.max(0, 1 - img - rep);
    bar.innerHTML =
      '<i class="mi" style="width:' + (img * 100).toFixed(1) + '%"></i>' +
      '<i class="mr" style="width:' + (rep * 100).toFixed(1) + '%"></i>' +
      '<i style="width:' + (ctx * 100).toFixed(1) + '%"></i>';
  }

  /* ---------- truth + reports ---------- */
  function renderTruth() {
    var t = TILES[state.tile];
    var pill = $('truthPill');
    pill.className = 'pill d' + t.truth.damage;
    pill.textContent = DN[t.truth.damage];
    $('truthBag').textContent = t.truth.bag
      ? 'a relevant report is present'
      : 'no relevant report';
    $('truthSitrep').textContent = t.truth.sitrep;
  }

  function renderReports() {
    var t = TILES[state.tile];
    var box = $('reports');
    $('repCount').textContent = t.n_reports === 0 ? '' : '(' + t.n_reports + ')';
    box.innerHTML = '';

    if (t.n_reports === 0) {
      var d = document.createElement('div');
      d.className = 'empty';
      d.innerHTML = '<b>No reports reached this tile.</b> Cloud cover, a night pass or a network blackout leaves the model with imagery alone. Seven of the sixteen test tiles look like this, so the architecture has to produce a report anyway.';
      box.appendChild(d);
      return;
    }

    var maxA = 0;
    t.reports.forEach(function (r) { if (r.alpha !== null && r.alpha > maxA) maxA = r.alpha; });

    t.reports.forEach(function (r, i) {
      var el = document.createElement('div');
      el.className = 'rep ' + (r.relevant ? 'rel' : 'dis');
      var dist = Math.sqrt(r.dx * r.dx + r.dy * r.dy).toFixed(2);
      var aHtml = '';
      if (r.alpha !== null) {
        var w = maxA > 0 ? (r.alpha / maxA) * 100 : 0;
        aHtml =
          '<div class="alpha"><span class="v">' + r.alpha.toFixed(3) + '</span>' +
          '<span class="alphabar"><i style="width:' + w.toFixed(0) + '%"></i></span></div>';
      }
      el.innerHTML =
        '<span class="ix">' + String(i + 1).padStart(2, '0') + '</span>' +
        '<div class="txt">' + esc(r.text) +
        '<div class="geo">' + dist + ' km from centre &middot; ' +
        (r.dt >= 0 ? '+' : '') + r.dt.toFixed(1) + ' h &middot; ' +
        (r.relevant ? 'about this tile' : 'unrelated') + '</div></div>' + aHtml;
      box.appendChild(el);
    });
  }

  /* ---------- variant comparison ---------- */
  var VMETA = {
    early: { name: 'Early fusion', kind: 'feature-level' },
    late: { name: 'Late fusion', kind: 'decision-level' },
    hybrid: { name: 'Hybrid fusion', kind: 'cross-attention' }
  };

  function counts(s) {
    var m = s.match(/(\d+)\s+destroyed\s+(\d+)\s+major damage\s+(\d+)\s+minor damage\s+(\d+)\s+undamaged/);
    if (!m) return null;
    return [m[1], m[2], m[3], m[4]];
  }

  function markSitrep(gen, truth) {
    var g = counts(gen);
    var r = counts(truth);
    if (!g || !r) return esc(gen);
    var seen = 0;
    return esc(gen).replace(/\b\d+\b/g, function (d) {
      if (seen >= 4) return d;
      var ok = d === r[seen];
      seen++;
      return '<span class="num ' + (ok ? 'hit' : 'miss') + '">' + d + '</span>';
    });
  }

  function renderVariants() {
    var t = TILES[state.tile];
    var box = $('variants');
    box.innerHTML = '';
    ['early', 'late', 'hybrid'].forEach(function (k) {
      var v = t.variants[k];
      var meta = VMETA[k];
      var dOk = v.damage === t.truth.damage;
      var bOk = v.bag === t.truth.bag;
      var el = document.createElement('div');
      el.className = 'vcard' + (k === 'hybrid' ? ' win' : '');
      el.innerHTML =
        '<div class="vhead"><span class="vname">' + meta.name + '</span>' +
        '<span class="vkind">' + meta.kind + '</span></div>' +
        '<div class="vbody">' +
        '<div class="sitrep">' + markSitrep(v.sitrep, t.truth.sitrep) + '</div>' +
        '<div class="vstats">' +
        '<span>damage <b>' + DN[v.damage] + '</b> <span class="mark ' + (dOk ? 'y' : 'n') + '">' + (dOk ? '✓' : '✗') + '</span></span>' +
        '<span>relevance <b>' + (v.bag ? 'yes' : 'no') + '</b> <span class="mark ' + (bOk ? 'y' : 'n') + '">' + (bOk ? '✓' : '✗') + '</span></span>' +
        '</div></div>';
      box.appendChild(el);
    });

    var note = $('tileNote');
    if (t.n_reports === 0) {
      note.innerHTML = '<b>This tile has no text.</b> All three fall back on imagery alone and correctly report that no relevant field report exists. The difference between the three mechanisms only appears once there is text to read.';
    } else {
      var agree = t.variants.early.damage === t.variants.hybrid.damage &&
        t.variants.late.damage === t.variants.hybrid.damage;
      note.innerHTML = agree
        ? '<b>All three agree on the damage class here</b> &mdash; as they do on every tile in the test set, because 95% carry the label &ldquo;destroyed&rdquo; and all three learned to answer it constantly. Look instead at the final clause: whether the model reports needs is where they come apart.'
        : '<b>The three disagree on this tile.</b> Compare which numbers each one got right, and remember all three read the identical frozen image and text features.';
    }
  }

  /* ---------- results bars ---------- */
  function bars(elId, rows, baseline, maxV) {
    var el = $(elId);
    el.innerHTML = '';
    rows.forEach(function (r) {
      var pct = (r.v / maxV) * 100;
      var basePct = baseline === null ? null : (baseline / maxV) * 100;
      var row = document.createElement('div');
      row.className = 'barrow';
      row.innerHTML =
        '<span class="nm">' + r.k + '</span>' +
        '<span class="track"><i class="' + (r.hi ? 'hi' : '') + '" style="width:' + pct.toFixed(1) + '%"></i>' +
        (basePct === null ? '' : '<span class="base" style="left:' + basePct.toFixed(1) + '%"></span>') +
        '</span>' +
        '<span class="vl">' + r.v.toFixed(2) + '</span>';
      el.appendChild(row);
    });
  }

  function renderResults() {
    // Report relevance. All three clear the always-say-yes baseline of 0.530
    // by a wide margin, and all three land on exactly the same score.
    bars('bars3', [
      { k: 'early', v: 0.97, hi: false },
      { k: 'late', v: 0.97, hi: false },
      { k: 'hybrid', v: 0.97, hi: true }
    ], 0.530, 1.0);

    // Same task with every report deleted. All three fall to 0.169, which is
    // exactly the always-say-no baseline: without text they cannot do it.
    bars('bars12', [
      { k: 'early', v: 0.17, hi: false },
      { k: 'late', v: 0.17, hi: false },
      { k: 'hybrid', v: 0.17, hi: true }
    ], 0.530, 1.0);

    // Does the GENERATED SITREP change when the reports are deleted?
    // Only cross-attention's does.
    bars('barsDrop', [
      { k: 'early', v: 0.00, hi: false },
      { k: 'late', v: 0.00, hi: false },
      { k: 'hybrid', v: 1.00, hi: true }
    ], null, 1.0);
  }

  /* ---------- plumbing ---------- */
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function select(i) {
    state.tile = i;
    state.word = 0;
    syncStrip();
    renderStage();
    renderTruth();
    renderReports();
    renderWords();
    renderMass();
    renderVariants();
  }

  function wireLabels() {
    var b = $('labTog');
    if (!b) return;
    b.addEventListener('click', function () {
      state.labels = !state.labels;
      b.setAttribute('aria-pressed', String(state.labels));
      renderIcons();
    });
  }

  function init() {
    wireLabels();
    buildStrip();
    var tabs = document.querySelectorAll('.imgtabs button');
    for (var i = 0; i < tabs.length; i++) {
      (function (btn) {
        btn.addEventListener('click', function () {
          state.layer = btn.dataset.layer;
          renderStage();
        });
      })(tabs[i]);
    }
    renderResults();
    select(state.tile);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
