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

export interface SnakeFrame {
  iteration: number;
  stage: number;
  stage_iteration: number;
  scale: number | null;
  contour: number[][];
  start_pixel: number[];
  chain_code: number[];
  moved_points: number;
  converged: boolean;
}

export interface SnakeMeasurements {
  converged: boolean;
  area: number;
  perimeter: number;
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

  async runIterations(
    apiUrl: string,
    file: File,
    contour: number[][],
    params: SnakeParams,
    onFrame: (frame: SnakeFrame) => Promise<void>,
  ): Promise<SnakeMeasurements> {
    const form = new FormData();
    form.append('file', file);
    form.append('contour', JSON.stringify(contour));
    form.append('alpha', String(params.alpha));
    form.append('beta', String(params.beta));
    form.append('gamma', String(params.gamma));
    form.append('scales', params.scales);
    form.append('max_iterations', String(params.maxIterations));
    form.append('num_points', String(params.numPoints));

    const response = await fetch(
      apiUrl.replace(/\/$/, '') + '/active-contour/iterations',
      { method: 'POST', body: form },
    );
    if (!response.ok) {
      let detail = `Request failed (${response.status}).`;
      try {
        const body = await response.json();
        detail = body.detail ?? detail;
      } catch {
        // Keep the HTTP status message if the server response is not JSON.
      }
      throw new Error(detail);
    }
    if (!response.body) {
      throw new Error('The backend did not provide a response stream.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    let finished = false;
    let measurements: SnakeMeasurements | null = null;

    const processLine = async (line: string) => {
      if (!line.trim()) return;
      const event = JSON.parse(line);
      if (event.type === 'error') {
        throw new Error(event.detail ?? 'The backend could not encode this contour.');
      }
      if (event.type === 'frame') {
        await onFrame(event as SnakeFrame);
      } else if (event.type === 'done') {
        if (
          typeof event.converged !== 'boolean'
          || typeof event.area !== 'number'
          || typeof event.perimeter !== 'number'
        ) {
          throw new Error('The backend returned invalid final contour measurements.');
        }
        finished = true;
        measurements = {
          converged: event.converged,
          area: event.area,
          perimeter: event.perimeter,
        };
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const lines = pending.split('\n');
      pending = lines.pop() ?? '';
      for (const line of lines) {
        await processLine(line);
      }
      if (done) break;
    }
    await processLine(pending);
    if (!finished) {
      throw new Error('The backend stream ended before the snake converged.');
    }
    if (!measurements) {
      throw new Error('The backend stream did not include final contour measurements.');
    }
    return measurements;
  }
}
