const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { EventEmitter } = require('node:events');
const source = fs.readFileSync(path.join(__dirname, '../Max MIDI Effect/W1_Hit/w1_hit.js'), 'utf8');

function harness(installed = true) {
  const handlers = {}, outlets = [], children = [], calls = [];
  const api = { post() {}, outlet: (message) => outlets.push(message), addHandler: (name, fn) => handlers[name] = fn };
  vm.runInNewContext(source, {
    __dirname: '/User Library/Max MIDI Effect/W1_Hit',
    process: { platform: 'darwin', env: {}, cwd: () => '/unrelated' },
    require(name) {
      if (name === 'max-api') return api;
      if (name === 'path') return path;
      if (name === 'fs') return { existsSync: (p) => !p.includes('.venv') || installed };
      if (name === 'child_process') return { spawn(command, args, options) {
        calls.push({ command, args, options });
        const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter();
        children.push(child); return child;
      }};
      throw Error(name);
    },
  });
  return { handlers, outlets, children, calls };
}

(async () => {
  const h = harness();
  const running = h.handlers.bang();
  await h.handlers.bang();
  assert.equal(h.children.length, 1, 'double click must not start two generators');
  h.children[0].stdout.emit('data', JSON.stringify({ hit: Array(128).fill(1), vel: Array(128).fill(100) }));
  h.children[0].emit('close', 0);
  await running;
  assert.equal(h.outlets.filter((x) => x[0] === 'clipcmd').length, 2);

  const invalid = h.handlers.bang();
  h.children[1].stdout.emit('data', '{"hit":[1],"vel":[500]}'); h.children[1].emit('close', 0);
  await invalid;
  assert.equal(h.outlets.filter((x) => x[0] === 'clipcmd').length, 2, 'bad output must not delete clip notes');
  const retry = h.handlers.bang(); h.children[2].emit('error', Error('failed')); await retry;
  assert.equal(h.children.length, 3, 'lock must clear after error');

  const setup = harness(false);
  assert.equal(setup.calls.length, 0, 'loading must not install software');
  setup.handlers.install_deps();
  assert.equal(setup.calls[0].command, '/bin/bash');
  assert.equal(setup.calls[0].args[0], '/User Library/Max MIDI Effect/W1_Hit/install/install_mac.command');
  assert.equal(setup.calls[0].options.shell, undefined, 'path with spaces must remain a single argument');
  setup.children[0].emit('error', Error('failed'));
  setup.handlers.install_deps();
  assert.equal(setup.calls.length, 2, 'installation must be retryable after spawn error');
  console.log('Node bridge regression checks passed');
})().catch((error) => { console.error(error); process.exitCode = 1; });
