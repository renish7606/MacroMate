import time


class RequestLoggingMiddleware:
    """
    Lightweight request logger for development.

    Prints a single line to the terminal for every incoming HTTP request,
    including method, path, status code, duration, and authenticated user.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start = time.time()
        user = request.user if hasattr(request, "user") else None
        username = getattr(user, "username", "anonymous") if user and user.is_authenticated else "anonymous"

        response = self.get_response(request)

        duration_ms = int((time.time() - start) * 1000)
        try:
            status = response.status_code
        except Exception:
            status = "?"

        print(
            f"[REQUEST] {request.method} {request.get_full_path()} "
            f"→ {status} ({duration_ms}ms) user={username}"
        )

        return response

