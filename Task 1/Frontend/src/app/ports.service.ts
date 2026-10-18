import { Injectable } from '@angular/core';
import { PortId, Op } from './ops';
export interface PortImage { blob: Blob; url: string; }
const API = '/api';

@Injectable({ providedIn: 'root' })
export class PortsService {
  images: Record<PortId, PortImage | null> = { input: null, output1: null, output2: null };
  active: PortId | null = null;

  set(id: PortId, blob: Blob) {
    const old = this.images[id];
    if (old) URL.revokeObjectURL(old.url);
    this.images[id] = { blob, url: URL.createObjectURL(blob) };
    this.active = id;
  }
  clear(id: PortId) {
    const old = this.images[id];
    if (old) URL.revokeObjectURL(old.url);
    this.images[id] = null;
    if (this.active === id) this.active = null;
  }

  async run(op: Op, files: Blob[], vals: any): Promise<Blob | any> {
    const fd = new FormData();
    if (op.dual) { fd.append('file1', files[0], 'a.png'); fd.append('file2', files[1], 'b.png'); }
    else fd.append('file', files[0], 'a.png');
    const v = { ...vals };
    if (op.id === 'hybrid') v.high_pass_coefficient = +(1 - v.low_pass_coefficient).toFixed(2), v.low_pass_coefficient = +(+v.low_pass_coefficient).toFixed(2);
    for (const k of Object.keys(v)) fd.append(k, String(v[k]));
    let res: Response;
    try { res = await fetch(API + op.endpoint, { method: 'POST', body: fd }); }
    catch { throw new Error('Cannot reach backend. Is it running on http://localhost:8000 ?'); }
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try { const j = await res.json(); const d = j.detail; msg = typeof d === 'string' ? d : JSON.stringify(d); } catch {}
      throw new Error(msg);
    }
    return op.kind === 'json' ? res.json() : res.blob();
  }
}
