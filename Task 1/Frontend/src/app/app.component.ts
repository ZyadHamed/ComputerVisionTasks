import { Component } from '@angular/core';
import { PortComponent } from './port.component';
import { OperationPanelComponent } from './operation-panel.component';
import { PORTS } from './ops';

@Component({
  selector: 'app-root', standalone: true, imports: [PortComponent, OperationPanelComponent],
  template: `
  <div class="layout">
    <aside><h1>Image Lab</h1><app-operation-panel /></aside>
    <main>
      <div class="dot"><i></i> PORTS</div>
      <div class="grid">@for (p of ports; track p.id) { <app-port [id]="p.id" [title]="p.title" /> }</div>
    </main>
  </div>`,
  styles: [`
  .layout{display:grid;grid-template-columns:360px 1fr;gap:20px;padding:20px;min-height:100vh;align-items:start}
  aside{position:sticky;top:20px;max-height:calc(100vh - 40px);overflow:auto}
  h1{margin:0 0 12px;font-size:20px;color:var(--dark)}
  .dot{font-weight:700;letter-spacing:.08em;font-size:12px;color:#5b6b85;margin:6px 0 12px;display:flex;gap:8px;align-items:center}
  .dot i{width:9px;height:9px;border-radius:50%;background:var(--ok)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
  @media(max-width:900px){.layout{grid-template-columns:1fr}aside{position:static;max-height:none}}`]
})
export class AppComponent { ports = PORTS; }
