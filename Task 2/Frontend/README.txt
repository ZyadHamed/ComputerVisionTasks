Needs Node.js 18.19+ (or 20+).

1. npm install
2. npm start
3. Open http://localhost:4200

The backend must be running (uvicorn Backend.main:app --reload).
If it is not on http://localhost:8000, change the "API URL" box.

Clicking Run Snake animates the contour at each iteration and updates its
Freeman chain code until each scale converges or reaches the iteration limit.
The output displays the final contour's area and chain-code perimeter.

Tip: use small images (around 512 px) - the snake is slow on big ones.
