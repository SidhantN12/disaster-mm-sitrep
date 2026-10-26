# Product concept console

A design concept for the deployed product, kept separate from the research
results on purpose. Nothing here is model output.

- `aerial.py` — procedural aerial imagery. Multi-octave value noise for ground
  and vegetation, a wandering river, a road grid, buildings with cast shadows
  and ridge lines, and a damage pass that darkens and rubbles each structure by
  level. Drawn, not photographed.
- `build_tiles.py` — composes nine sectors with damage profiles, field reports,
  SITREPs and priority scores into `vision_data.json`.
- `console_page.html` — the console template; the data is inlined at build time.

Build:

    python build_tiles.py
    python -c "import pathlib; \
      page=pathlib.Path('console_page.html').read_text(); \
      data=pathlib.Path('vision_data.json').read_text().replace('</','<\\\\/'); \
      pathlib.Path('../results/triage_concept.html').write_text( \
        page.replace('<script>\n(function () {', \
        '<script>window.__VISION__='+data+';</script>\n<script>\n(function () {',1))"

The imagery and assessments are illustrative. Measured results from the trained
model are in `results/fusion_results.json` and reported by
`results/sitrep_demo.html`.
