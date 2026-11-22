import { Component } from '@angular/core';
import { FeatureWorkspaceComponent } from './feature-workspace.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FeatureWorkspaceComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {}
