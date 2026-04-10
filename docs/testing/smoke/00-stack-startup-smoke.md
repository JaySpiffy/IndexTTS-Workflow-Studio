# Stack Startup Smoke

## Purpose

Confirm the Docker stack starts cleanly and the app reaches a usable ready state before you test any tab.

## Preconditions

- Docker Desktop is running
- NVIDIA runtime is available if you expect GPU mode
- model files already exist in `shared/models/checkpoints`, or you are prepared to wait for first-start download

## Steps

1. Start the app with [../../docker/start.bat](../../docker/start.bat).
2. Wait for the frontend link to appear.
3. Open [http://localhost:3000](http://localhost:3000).
4. If the page shows a temporary backend error, wait for model load to finish and refresh once.
5. Check [http://localhost:3000/api/health](http://localhost:3000/api/health).
6. Check [http://localhost:8001/health](http://localhost:8001/health).
7. Confirm the header badge in the UI changes from `Checking API...` to a connected state.

## Expected Results

- frontend loads without a blank page
- `api/health` returns JSON
- backend `health` returns JSON
- `model_loaded` is `true`
- if GPU is available, the header shows `GPU: cuda:0 + DeepSpeed` or the actual active CUDA device
- if GPU is not available, the header still shows connected status and CPU mode cleanly

## Common Failure Signs

- `502 Bad Gateway` that never clears after startup
- header remains `API Disconnected`
- repeated `Failed to fetch` toasts after the backend is healthy
- backend health says `model_loaded: false`
- unexpected CPU fallback when you expected GPU

## Notes

- A short temporary `502` during model load is acceptable on cold startup.
- Do one `Ctrl+F5` if you suspect a stale frontend bundle after a rebuild.
