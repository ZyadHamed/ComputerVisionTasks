import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface SnakeParams {
  alpha: number;
  beta: number;
  gamma: number;
  scales: string;
  maxIterations: number;
  numPoints: number;
}

@Injectable({ providedIn: 'root' })
export class SnakeService {
  constructor(private http: HttpClient) {}

  // Sends the image + contour to the backend, returns the result PNG.
  runOverlay(apiUrl: string, file: File, contour: number[][], p: SnakeParams): Observable<Blob> {
    const form = new FormData();
    form.append('file', file);
    form.append('contour', JSON.stringify(contour));
    form.append('alpha', String(p.alpha));
    form.append('beta', String(p.beta));
    form.append('gamma', String(p.gamma));
    form.append('scales', p.scales);
    form.append('max_iterations', String(p.maxIterations));
    form.append('num_points', String(p.numPoints));

    const url = apiUrl.replace(/\/$/, '') + '/active-contour/overlay';
    return this.http.post(url, form, { responseType: 'blob' });
  }
}
