import { Component, Input } from '@angular/core';
import { PortsService } from './ports.service';
import { PortId } from './ops';

@Component({
  selector: 'app-port', standalone: true,
  template: `
  <section class="port" [class.active]="svc.active === id">
    <header>
      <span class="t">{{ title }}</span>
      @if (svc.active === id) { <span class="badge">ACTIVE</span> }
      <span class="sp"></span>
      @if (id === 'input') {
        <label class="btn sm">Upload<input type="file" accept="image/*" hidden (change)="pick($event)"></label>
      }
      @if (svc.images[id]) {
        <button class="btn sm" (click)="download()">Download</button>
        <button class="btn sm" (click)="svc.clear(id)">Clear</button>
      }
    </header>
    <div class="view" (dragover)="$event.preventDefault()" (drop)="drop($event)">
      @if (svc.images[id]; as img) {
        <img [src]="img.url" (load)="dim($event)" alt="{{title}}">
      } @else {
        <div class="empty">{{ id === 'input' ? 'Drop an image here or click Upload' : 'Empty — apply an operation to send a result here' }}</div>
      }
    </div>
    <footer>{{ size }}</footer>
  </section>`,
  styles: [`
  .port{background:var(--panel);border:1px solid var(--border);border-radius:12px;overflow:hidden;display:flex;flex-direction:column;min-width:0}
  .port.active{border:2px solid var(--primary);box-shadow:0 0 0 3px rgba(26,115,232,.12)}
  header{display:flex;align-items:center;gap:8px;padding:10px 12px;background:var(--head)}
  .t{font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12px;color:#5b6b85}
  .badge{background:var(--ok);color:#fff;font-size:10px;font-weight:800;padding:2px 8px;border-radius:5px}
  .sp{flex:1}.sm{padding:4px 10px;font-size:12px;cursor:pointer}
  .view{background:#8f8f8f;min-height:300px;flex:1;display:flex;align-items:center;justify-content:center;overflow:auto}
  img{max-width:100%;max-height:420px;display:block}
  .empty{color:#e8edf5;text-align:center;padding:24px;font-size:13px}
  footer{padding:6px 12px;font-size:12px;color:var(--muted);min-height:28px}`]
})
export class PortComponent {
  @Input() id!: PortId; @Input() title = '';
  size = '';
  constructor(public svc: PortsService) {}
  pick(e: Event) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) this.svc.set(this.id, f); (e.target as HTMLInputElement).value = ''; }
  drop(e: DragEvent) { e.preventDefault(); const f = e.dataTransfer?.files?.[0]; if (f && f.type.startsWith('image/')) this.svc.set(this.id, f); }
  dim(e: Event) { const i = e.target as HTMLImageElement; this.size = `${i.naturalWidth} × ${i.naturalHeight} px`; }
  download() {
    const img = this.svc.images[this.id]; if (!img) return;
    const a = document.createElement('a'); a.href = img.url; a.download = `${this.id}.png`; a.click();
  }
}
