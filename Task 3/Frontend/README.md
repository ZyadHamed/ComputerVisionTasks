# Vision Lab — Feature matching, large-input update

This Angular 19 workspace keeps the **Keypoints** and **Feature matching** tabs and removes Task 2. This version must be deployed together with the matching `Backend/main.py` and feature services supplied in the backend package.

## What was fixed

- **Harris / MOPS HTTP 422**: `features1` and `features2` are now multipart `application/json` **file** parts (`features1.json`, `features2.json`), matching FastAPI `UploadFile = File(...)`. The previous frontend sent enormous form strings and could get `Expected UploadFile, received: <class 'str'>` or `Part exceeded maximum size of 1024KB`.
- The UI does not send `max_keypoints` or `max_matches`; the updated backend defaults both to unlimited. There is no 500/4000 hard feature cap.
- No frontend checks reject files based on their byte count or pixel dimensions. The original image Blob is uploaded unchanged; the backend no longer rejects uploads over 25 MiB or images over 16 megapixels.
- Matching returns **all** pairs that pass the chosen SSD/NCC ratio/cross-check/absolute-score filters. There is no 80-line drawing limit. Interactive lines/circles render in animation-frame batches, without dropping points. The results table is paginated 100 rows per page solely to keep DOM rendering responsive.
- For very large images, *only the browser canvas preview* is scaled to fit browser graphics memory. Original uploads and backend computation use the **full image resolution**. The `Download original` control preserves the uploaded bytes; the `API PNG` route can produce a full-resolution server-rendered match image.
- If the browser cannot decode an image for preview, the original blob is still kept and can be sent for processing.
- FastAPI validation errors no longer dump the full input descriptor JSON into the error banner.

## Start

```bash
npm install
npm start
```

Open <http://localhost:4200> with FastAPI listening on `http://localhost:8000`. Change `apiUrl` in `src/app/feature-lab.service.ts` if needed.

## Backend compatibility — update BOTH projects

The corresponding backend changes the signatures of `/features/match` and `/features/match/visualize` to:

```python
features1: UploadFile = File(...),
features2: UploadFile = File(...),
```

and reads them with `json.loads(await features1.read())` and `json.loads(await features2.read())`. Install the included patched backend files before using this Angular version. Mixing an old frontend (JSON text) and new backend (JSON file), or vice versa, causes HTTP 422 errors. Both image fields (`image1`, `image2`) remain normal uploaded files.

The backend uses the same endpoints as before:

| Endpoint | Use |
|---|---|
| `GET /images/latest` | Load last saved image |
| `POST /images/upload` | Save image |
| `POST /corners` | Harris/Lambda Minus with MOPS/SIFT descriptors |
| `POST /corners/overlay` | Keypoint-circle PNG |
| `POST /features/sift` | Full SIFT extraction |
| `POST /features/sift/visualize` | Full SIFT keypoint overlay PNG |
| `POST /features/match` | SSD/NCC match JSON |
| `POST /features/match/visualize` | Matched image PNG |

The corner endpoints have `save_as_latest=False` by default in the patch, so extracting two images no longer overwrites the shared latest-image slot.

## Tests

```bash
npm run check:static
npm run test:contracts
# Once dependencies are installed:
npm run build
```

Static TypeScript checks and all eight request-contract tests passed during creation, including JSON file parts larger than 1 MiB. Angular's full CLI build and browser-level testing could not be run here without installing Angular npm dependencies.

**Resource note:** No application-level image-size or feature-count caps remain, but physical RAM, available disk, browser decoding capabilities, host request limits, and processing time still apply. Exact matching must compare image A features against image B features; its time cost still grows with `N1 × N2`, though the backend no longer allocates the full pairwise score matrix in RAM.
