import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const children = [];
let closing = false;
const close = () => {
  if (closing) return;
  closing = true;
  for (const child of children) child.kill();
};
process.on('SIGINT', close);
process.on('SIGTERM', close);
process.on('exit', close);

async function healthy(url, expected) {
  try { const response = await fetch(url, { signal: AbortSignal.timeout(1500) }); return response.ok && expected(await response.text()); }
  catch { return false; }
}

function launch(program, args) {
  const child = spawn(program, args, { cwd: root, stdio: 'inherit', windowsHide: true });
  children.push(child);
  child.on('error', error => { console.error(error.message); close(); process.exitCode = 1; });
  child.on('exit', code => { if (!closing) { close(); process.exitCode = code || 1; } });
}

if (!await healthy('http://127.0.0.1:8788/api/status', text => { try { return typeof JSON.parse(text).connected === 'boolean'; } catch { return false; } })) {
  launch(path.join(root, '.venv', 'Scripts', 'python.exe'), ['-m', 'uvicorn', 'backend.server:app', '--host', '127.0.0.1', '--port', '8788']);
}
if (!await healthy('http://127.0.0.1:5173/', text => text.includes('Vector Lab'))) {
  launch(process.execPath, [path.join(root, 'scripts', 'run-framework.mjs'), 'dev']);
}
console.log('\nVector Lab: http://127.0.0.1:5173/\nLocal model bridge: http://127.0.0.1:8788\nKeep this terminal open. Ctrl+C stops only processes started by this launcher.\n');
