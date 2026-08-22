// Guardia anti-drift de la clave canonica del Laboratorio (panel.html).
//
// canon() en Python (panel.py) y canonKey() en JS (embebido en panel.html)
// DEBEN producir la misma clave para la misma config: si driftean, el panel
// deja de reconocer que una config compuesta en la UI ya fue corrida.
//
// Este test carga el panel.html GENERADO, extrae su <script>, lo evalua con
// stubs minimos de DOM y re-verifica cada clave embebida (las claves del
// payload las genero Python; canonKey es el JS real del panel).
//
//   node eda/test_canon_panel.js
//
// Ademas verifica que las claves de la suite sean unicas (dos experimentos
// distintos no pueden canonicalizar igual).
'use strict';
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'panel.html'), 'utf8');
const inicio = html.indexOf("const DATA = ");
const fin = html.lastIndexOf('</script>');
if (inicio < 0 || fin < 0) throw new Error('no encuentro el <script> del panel');
let js = html.slice(inicio, fin);

// stubs de DOM/navegador: solo queremos DATA + las funciones puras
const elemento = () => new Proxy({style: {}, classList: {toggle(){}, add(){}, remove(){}},
  dataset: {}, value: '', checked: false, textContent: '', innerHTML: ''},
  {get: (t, p) => (p in t ? t[p] : (typeof p === 'string' && p.startsWith('on') ? null : t[p])),
   set: () => true});
const sandbox = {
  document: {getElementById: elemento, querySelectorAll: () => [], addEventListener(){},
             body: elemento()},
  window: {addEventListener(){}, scrollTo(){}},
  navigator: {},
  location: {hash: ''},
  console,
};
// cortar el script despues de canonKey + self-test para no evaluar el wiring de UI
const corte = js.indexOf('const suiteByKey');
if (corte < 0) throw new Error('no encuentro el marcador suiteByKey');
js = js.slice(0, corte);

const vm = require('vm');
const ctx = vm.createContext(sandbox);
vm.runInContext(js, ctx);
// const/let del script no cuelgan del global del sandbox: pedirlos al contexto
const DATA = vm.runInContext('DATA', ctx);
const canonKey = vm.runInContext('canonKey', ctx);

let mal = 0, total = 0;
for (const [nombre, v] of Object.entries(DATA.suite)) {
  total++;
  if (canonKey(v.cfg) !== v.key) { mal++; console.error('suite drift:', nombre); }
}
for (const [key, g] of Object.entries(DATA.results)) {
  total++;
  if (canonKey(g.cfg) !== key) { mal++; console.error('resultado drift:', g.nombre || key.slice(0, 60)); }
}
const claves = Object.values(DATA.suite).map(v => v.key);
const unicas = new Set(claves);
if (unicas.size !== claves.length) {
  mal++;
  console.error(`claves de suite NO unicas: ${claves.length} configs, ${unicas.size} claves`);
}
if (mal) {
  console.error(`FALLO: ${mal} problema(s) sobre ${total} claves`);
  process.exit(1);
}
console.log(`OK: ${total} claves verificadas (Python == JS), ${unicas.size} claves de suite unicas`);
