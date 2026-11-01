import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

export interface DetectedLine {
  rho: number;
  theta: number;
  theta_degrees: number;
}

export interface DetectedCircle {
  radius: number;
  center_x: number;
  center_y: number;
  votes: number;
}

export interface DetectedEllipse {
  center_x: number;
  center_y: number;
  a: number;
  b: number;
  alpha: number;
  alpha_degrees: number;
  votes: number;
}

export interface LinesResponse {
  count: number;
  lines: DetectedLine[];
  accumulator_shape: number[];
}

export interface CirclesResponse {
  count: number;
  circles: DetectedCircle[];
  accumulator_shape: number[];
}

export interface EllipsesResponse {
  count: number;
  edge_points: number;
  ellipses: DetectedEllipse[];
}

export type FormOptions = Record<string, string | number | boolean | null | undefined>;

@Injectable({ providedIn: 'root' })
export class ImageProcessingService {
  // Change to your backend host when the UI and API aren't on the same computer.
  readonly apiUrl = 'http://localhost:8000';

  constructor(private readonly http: HttpClient) {}

  async getLatestImage(): Promise<Blob | null> {
    try {
      return await firstValueFrom(
        this.http.get(`${this.apiUrl}/images/latest`, { responseType: 'blob' }),
      );
    } catch (error) {
      // The API returns 400 when no image has been uploaded yet.
      if (error instanceof HttpErrorResponse && error.status === 400) return null;
      throw error;
    }
  }

  async uploadLatestImage(file: File): Promise<void> {
    const form = new FormData();
    form.append('file', file, file.name);
    await firstValueFrom(this.http.post(`${this.apiUrl}/images/upload`, form));
  }

  private makeForm(image: Blob, options: FormOptions): FormData {
    const form = new FormData();
    // Send raw input pixels, never canvas annotations. Chaining is non-destructive.
    form.append('file', image, 'workspace-source.png');
    // Requires the small backend compatibility patch bundled with this ZIP.
    // Keeps the globally saved original image unchanged when processing outputs.
    form.append('save_as_latest', 'false');
    for (const [key, value] of Object.entries(options)) {
      if (value !== null && value !== undefined && value !== '') {
        form.append(key, String(value));
      }
    }
    return form;
  }

  runCanny(image: Blob, options: FormOptions): Promise<Blob> {
    return firstValueFrom(this.http.post(`${this.apiUrl}/canny`, this.makeForm(image, options), {
      responseType: 'blob',
    }));
  }

  detectLines(image: Blob, options: FormOptions): Promise<LinesResponse> {
    return firstValueFrom(this.http.post<LinesResponse>(
      `${this.apiUrl}/hough-lines`, this.makeForm(image, options),
    ));
  }

  detectCircles(image: Blob, options: FormOptions): Promise<CirclesResponse> {
    return firstValueFrom(this.http.post<CirclesResponse>(
      `${this.apiUrl}/hough-circles`, this.makeForm(image, options),
    ));
  }

  detectEllipses(image: Blob, options: FormOptions): Promise<EllipsesResponse> {
    return firstValueFrom(this.http.post<EllipsesResponse>(
      `${this.apiUrl}/hough-ellipses`, this.makeForm(image, options),
    ));
  }
}
