# Image Lab (Angular 17 + TypeScript)
1. Start backend: `uvicorn Backend.main:app --port 8000` (adjust module path to yours)
2. `npm install` then `npm start` -> http://localhost:4200
Requests go to `/api/*`, proxied to http://localhost:8000 (see proxy.conf.json) so no CORS setup is needed.
Three ports: Input, Output 1, Output 2. Pick a source port, an operation, tune parameters, choose destination (Output 1/2), Apply.
