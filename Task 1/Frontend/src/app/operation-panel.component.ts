import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { OPS, Op, PORTS, PortId } from './ops';
import { PortsService } from './ports.service';

@Component({
  selector: 'app-operation-panel', standalone: true, imports: [FormsModule],
  template: `
  <div class="card">
    <h3>Operation</h3>
    <div class="lbl">Source</div>
    <select [(ngModel)]="src">@for (p of ports; track p.id) { <option [value]="p.id">{{ p.title }}</option> }</select>

    @if (op.dual) {
      <div class="lbl">Second image (source 2)</div>
      <select [(ngModel)]="src2">@for (p of ports; track p.id) { <option [value]="p.id">{{ p.title }}</option> }</select>
    }

    <div class="lbl">Operation</div>
    <select [(ngModel)]="opId" (ngModelChange)="reset()">@for (o of ops; track o.id) { <option [value]="o.id">{{ o.label }}</option> }</select>
    @if (op.note) { <p class="note">{{ op.note }}</p> }

    @for (p of op.params; track p.key) {
      @if (!p.showIf || p.showIf(vals)) {
        @switch (p.type) {
          @case ('range') {
            <div class="lbl">{{ p.label }}</div>
            <div class="row">
              <input type="range" [min]="p.min" [max]="p.max" [step]="p.step" [(ngModel)]="vals[p.key]">
              <input type="number" class="num" [min]="p.min" [max]="p.max" [step]="p.step" [(ngModel)]="vals[p.key]">
            </div>
          }
          @case ('select') {
            <div class="lbl">{{ p.label }}</div>
            <select [(ngModel)]="vals[p.key]">@for (o of p.options; track o) { <option [value]="o">{{ o }}</option> }</select>
          }
          @case ('bool') {
            <label class="chk"><input type="checkbox" [(ngModel)]="vals[p.key]"> {{ p.label }}</label>
          }
        }
      }
    }

    <div class="lbl">Send result to</div>
    <div class="seg">
      @for (p of outs; track p.id) {
        <button type="button" [class.on]="dst === p.id" (click)="dst = p.id">{{ p.title }}</button>
      }
    </div>

    <hr>
    <button class="btn primary" [disabled]="busy" (click)="apply()">{{ busy ? 'Processing…' : '→ Apply' }}</button>
    @if (error) { <p class="err">{{ error }}</p> }
    @if (json) {
      <div class="lbl">JSON result</div>
      <pre>{{ json }}</pre>
      <button class="btn" (click)="saveJson()">Download JSON</button>
    }
  </div>`,
  styles: [`
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px}
  h3{margin:0 0 4px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#5b6b85}
  .row{display:flex;gap:10px;align-items:center}.num{width:90px!important;flex:none}
  .chk{display:flex;gap:8px;align-items:center;margin-top:12px;cursor:pointer}
  .seg{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
  .seg button{flex:1;padding:8px;border:0;background:#fff;color:var(--muted);font-weight:600;cursor:pointer}
  .seg button.on{background:var(--primary-soft);color:var(--primary)}
  hr{border:0;border-top:1px solid var(--border);margin:16px 0}
  .note{font-size:12px;color:var(--muted);margin:6px 0 0}
  .err{color:var(--err);font-size:13px;word-break:break-word}
  pre{max-height:220px;overflow:auto;background:var(--head);padding:8px;border-radius:8px;font-size:11px}`]
})
export class OperationPanelComponent {
  ops = OPS; ports = PORTS; outs = PORTS.filter(p => p.id !== 'input');
  src: PortId = 'input'; src2: PortId = 'output1'; dst: PortId = 'output1';
  opId = 'grayscale'; vals: any = {}; busy = false; error = ''; json = '';
  constructor(private svc: PortsService) { this.reset(); }
  get op(): Op { return OPS.find(o => o.id === this.opId)!; }
  reset() { this.vals = {}; for (const p of this.op.params) this.vals[p.key] = p.def; this.error = ''; this.json = ''; }

  async apply() {
    this.error = ''; this.json = '';
    const a = this.svc.images[this.src];
    if (!a) { this.error = `The source (${this.src}) has no image.`; return; }
    if (this.op.kind === 'copy') { this.svc.set(this.dst, a.blob); return; }
    const files = [a.blob];
    if (this.op.dual) {
      const b = this.svc.images[this.src2];
      if (!b) { this.error = `Second source (${this.src2}) has no image.`; return; }
      files.push(b.blob);
    }
    this.busy = true;
    try {
      const res = await this.svc.run(this.op, files, this.vals);
      if (this.op.kind === 'json') this.json = JSON.stringify(res, null, 1);
      else this.svc.set(this.dst, res);
    } catch (e: any) { this.error = e.message || String(e); }
    finally { this.busy = false; }
  }
  saveJson() {
    const url = URL.createObjectURL(new Blob([this.json], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = 'histogram.json'; a.click(); URL.revokeObjectURL(url);
  }
}
