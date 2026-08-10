/**
 * Write a real .npz in the browser.
 *
 * A .npz is a zip of .npy members, so with store-mode (no compression) both
 * formats are short enough to emit by hand. The point is that a clip pulled
 * off this page loads with np.load and matches the schema
 * motion_generator/infer.py already writes -- t, qpos, light01, rgb, affect,
 * dt_ms, generator -- so it drops straight into the repo's own tooling.
 */

const enc = new TextEncoder();

function npyHeader(dtype, shape) {
  const shapeStr = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
  let dict = `{'descr': '${dtype}', 'fortran_order': False, 'shape': ${shapeStr}, }`;
  // total header must be a multiple of 64 bytes, terminated by \n
  const base = 10 + dict.length + 1;
  const pad = (64 - (base % 64)) % 64;
  dict += " ".repeat(pad) + "\n";
  const out = new Uint8Array(10 + dict.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);   // \x93NUMPY v1.0
  new DataView(out.buffer).setUint16(8, dict.length, true);
  out.set(enc.encode(dict), 10);
  return out;
}

function npy(dtype, shape, data) {
  const head = npyHeader(dtype, shape);
  const body = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const out = new Uint8Array(head.length + body.length);
  out.set(head, 0);
  out.set(body, head.length);
  return out;
}

/** CRC-32, needed by the zip container. */
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[i] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** members: [{ name, bytes }] -> a store-mode zip blob. */
function zipStore(members) {
  const chunks = [];
  const central = [];
  let offset = 0;

  for (const m of members) {
    const name = enc.encode(m.name);
    const crc = crc32(m.bytes);
    const local = new Uint8Array(30 + name.length);
    const dv = new DataView(local.buffer);
    dv.setUint32(0, 0x04034b50, true);
    dv.setUint16(4, 20, true);           // version needed
    dv.setUint16(6, 0, true);            // flags
    dv.setUint16(8, 0, true);            // method 0 = store
    dv.setUint16(10, 0, true);           // time
    dv.setUint16(12, 0, true);           // date
    dv.setUint32(14, crc, true);
    dv.setUint32(18, m.bytes.length, true);
    dv.setUint32(22, m.bytes.length, true);
    dv.setUint16(26, name.length, true);
    dv.setUint16(28, 0, true);
    local.set(name, 30);
    chunks.push(local, m.bytes);

    const cen = new Uint8Array(46 + name.length);
    const cv = new DataView(cen.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, 0, true);
    cv.setUint16(14, 0, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, m.bytes.length, true);
    cv.setUint32(24, m.bytes.length, true);
    cv.setUint16(28, name.length, true);
    cv.setUint32(42, offset, true);
    cen.set(name, 46);
    central.push(cen);

    offset += local.length + m.bytes.length;
  }

  const cenSize = central.reduce((a, c) => a + c.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, members.length, true);
  ev.setUint16(10, members.length, true);
  ev.setUint32(12, cenSize, true);
  ev.setUint32(16, offset, true);

  return new Blob([...chunks, ...central, end], { type: "application/zip" });
}

/**
 * Pack a clip in the repo's own generated-clip schema.
 * `frames` is T x 9 in physical units; `affect` an 11-vector; `meta` the
 * provenance blob that ends up in the `generator` field.
 */
export function clipToNpz(frames, affect, meta) {
  const T = frames.length;
  const t = new Float32Array(T);
  const qpos = new Float32Array(T * 5);
  const light = new Float32Array(T);
  const rgb = new Uint8Array(T * 3);
  for (let i = 0; i < T; i++) {
    t[i] = i * 0.033;
    for (let j = 0; j < 5; j++) qpos[i * 5 + j] = frames[i][j];
    light[i] = frames[i][5];
    for (let j = 0; j < 3; j++) {
      rgb[i * 3 + j] = Math.max(0, Math.min(255, Math.round(frames[i][6 + j] * 255)));
    }
  }
  // np.array(str) is dtype '<U<n>': n UCS-4 codepoints, little-endian, so
  // np.load gives back a str rather than bytes -- same as infer.py writes
  const text = JSON.stringify(meta);
  const cps = Array.from(text);
  const gen = new Uint32Array(cps.length);
  cps.forEach((ch, i) => { gen[i] = ch.codePointAt(0); });

  return zipStore([
    { name: "t.npy", bytes: npy("<f4", [T], t) },
    { name: "qpos.npy", bytes: npy("<f4", [T, 5], qpos) },
    { name: "light01.npy", bytes: npy("<f4", [T], light) },
    { name: "rgb.npy", bytes: npy("|u1", [T, 3], rgb) },
    { name: "affect.npy", bytes: npy("<f4", [affect.length], Float32Array.from(affect)) },
    { name: "dt_ms.npy", bytes: npy("<i8", [], new BigInt64Array([33n])) },
    { name: "generator.npy", bytes: npy(`<U${gen.length}`, [], gen) },
  ]);
}

export function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
