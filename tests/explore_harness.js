// Runs the script of an explore.html page against a stub DOM + Plotly (node, no browser) and simulates a
// user: lasso the first 40 % of the particles -> group "good"; lasso 20-60 % -> group "other" (moves the
// overlap); undo + redo; export. Prints {first draw, note, exported groups file} as JSON for the pytest side.
'use strict';
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const data = html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const app = html.slice(html.lastIndexOf('<script>') + 8, html.lastIndexOf('</script>'));

const elements = {}, handlers = {}, draws = [], blobs = [];
function mk(tag, id) {
  return { tag: tag, id: id, style: {}, children: [], value: '', textContent: '', innerHTML: '', className: '',
    appendChild(c) { this.children.push(c); if (this.tag === 'select' && this.value === '') { this.value = c.value; } return c; },
    querySelector() { return mk('tbody'); }, on(ev, fn) { handlers[ev] = fn; }, remove() {}, click() {} };
}
global.document = {
  getElementById(id) {
    if (id === 'data') { return { textContent: data }; }
    if (!elements[id]) { elements[id] = mk(['axes', 'pcx', 'pcy', 'colour'].includes(id) ? 'select' : 'div', id); }
    return elements[id];
  },
  createElement(tag) { return mk(tag); },
  body: { appendChild() {} },
};
global.window = { matchMedia() { return { matches: false }; } };
global.Plotly = { react(id, traces, layout) { draws.push({ traces: traces.length, names: traces.map(t => t.name || ''), n: traces.reduce((s, t) => s + t.x.length, 0) }); } };
global.Blob = function (parts) { this.text = parts.join(''); blobs.push(this); };
global.URL = { createObjectURL() { return 'blob:x'; }, revokeObjectURL() {} };
global.prompt = function () { return null; };

eval(app);

const el = id => document.getElementById(id);
const select = ids => handlers.plotly_selected({ points: ids.map(i => ({ customdata: i })) });
const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
const first = draws[0];
const N = JSON.parse(data).names.length, a = Math.floor(0.2 * N), b = Math.floor(0.4 * N), c = Math.floor(0.6 * N);
select(range(0, b)); el('gname').value = 'good'; el('add').onclick();
select(range(a, c)); el('gname').value = 'other'; el('add').onclick();
const note = el('note').textContent;
el('undo').onclick();
const afterUndo = el('rest').textContent;
select(range(a, c)); el('gname').value = 'other'; el('add').onclick();
el('setname').value = 'my set'; el('export').onclick();
el('axes').value = 'PC1 / PC2'; el('axes').onchange();
console.log(JSON.stringify({ moved: b - a, first: first, note: note, afterUndo: afterUndo, last: draws[draws.length - 1],
  colour: el('colour').value, exported: JSON.parse(blobs[blobs.length - 1].text), pcpick: el('pcpick').style.display }));
