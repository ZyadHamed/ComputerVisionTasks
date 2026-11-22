declare module '@angular/core' {
  export function Component(arg: any): ClassDecorator;
  export function Injectable(arg: any): ClassDecorator;
  export function ViewChild(name: any): PropertyDecorator;
  export class ElementRef<T> { nativeElement: T; }
  export interface AfterViewInit { ngAfterViewInit(): void; }
  export interface OnDestroy { ngOnDestroy(): void; }
}
declare module '@angular/common' { export class DecimalPipe {} }
declare module '@angular/forms' { export class FormsModule {} }
declare module 'rxjs' {
  export interface Observable<T> { subscribe: any; }
  export function firstValueFrom<T>(source: Observable<T>): Promise<T>;
}
declare module '@angular/common/http' {
  import { Observable } from 'rxjs';
  export class HttpErrorResponse extends Error { status: number; error: any; }
  export interface HttpResponse<T> { body: T | null; headers: {get(name: string): string | null}; }
  export class HttpClient {
    get(url: string, options: { responseType: 'blob' }): Observable<Blob>;
    post<T>(url: string, body: any): Observable<T>;
    post(url: string, body: any, options: {responseType: 'blob'}): Observable<Blob>;
    post(url: string, body: any, options: {responseType: 'blob', observe: 'response'}): Observable<HttpResponse<Blob>>;
  }
  export function provideHttpClient(): any;
}
declare module '@angular/platform-browser' { export function bootstrapApplication(component: any, options: any): Promise<any>; }
