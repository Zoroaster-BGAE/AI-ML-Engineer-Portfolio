from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from dotenv import dotenv_values  # pip install python-dotenv


class ClientCreator:
    """
    Resolve a URL + SERVICE key pair by searching in this order:
      1) Existing process environment (os.environ)
      2) .env.secrets or .env in the current directory
      3) .env.secrets or .env in the parent directory

    Never prints or exposes secret *values* — only the *sources* they came from.
    """

    def __init__(
        self,
        url_key: str,
        service_key: str,
        filenames: Tuple[str, ...] = (".env.secrets", ".env"),
        start_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        url_key : str
            Env var name for the service URL (e.g., "DB_API_URL").
        service_key : str
            Env var name for the service key (e.g., "DB_API_SERVICE_KEY").
        filenames : tuple[str,...]
            Files to consider (in order) at each location.
        start_dir : Path | None
            Where to start searching for files. Defaults to Path.cwd().
        verbose : bool
            If True, prints a summary of where credentials were found.
        """
        self.url_key = url_key
        self.service_key = service_key
        self.filenames = filenames
        self.start_dir = Path(start_dir) if start_dir else Path.cwd()
        self.verbose = verbose

        self._sources: Dict[str, str] = {}  # e.g., {"DB_API_URL": "env", "DB_API_SERVICE_KEY": "/path/.env.secrets"}

    def _try_env(self) -> None:
        """Check existing environment first (no mutation)."""
        if os.getenv(self.url_key):
            self._sources[self.url_key] = "environment"
        if os.getenv(self.service_key):
            self._sources[self.service_key] = "environment"

    def _load_from_file_if_needed(self, file_path: Path) -> None:
        """
        Load only if at least one required key is missing AND present in the file.
        Does NOT overwrite existing environment variables.
        """
        if self.url_key in self._sources and self.service_key in self._sources:
            return  # already resolved both from env

        if not file_path.exists() or not file_path.is_file():
            return

        kv = dotenv_values(str(file_path))  # returns dict without touching os.environ
        need_url = self.url_key not in self._sources and self.url_key in kv and kv[self.url_key]
        need_key = self.service_key not in self._sources and self.service_key in kv and kv[self.service_key]

        # Only set the ones we still need
        if need_url:
            os.environ.setdefault(self.url_key, kv[self.url_key])  # don't overwrite if somehow set
            self._sources[self.url_key] = str(file_path)

        if need_key:
            os.environ.setdefault(self.service_key, kv[self.service_key])
            self._sources[self.service_key] = str(file_path)

    def _search_files(self) -> None:
        """
        After environment check, look in:
          - current directory
          - parent directory
        For each location, check the listed filenames in order.
        """
        locations = [self.start_dir, self.start_dir.parent]
        for loc in locations:
            for fname in self.filenames:
                if self.url_key in self._sources and self.service_key in self._sources:
                    return
                self._load_from_file_if_needed(loc / fname)

    def resolve(self) -> Tuple[str, str]:
        """
        Resolve and return (url, service_key). Raises an error if missing.

        Returns
        -------
        (url, service_key) : tuple[str, str]
        """
        # Step 1: env
        self._try_env()

        # Step 2 & 3: files in current and parent dirs
        if self.url_key not in self._sources or self.service_key not in self._sources:
            self._search_files()

        missing = [v for v in (self.url_key, self.service_key) if v not in self._sources]
        if missing:
            searched = [
                "environment",
                *(str(self.start_dir / f) for f in self.filenames),
                *(str(self.start_dir.parent / f) for f in self.filenames),
            ]
            msg = (
                f"Missing credentials: {', '.join(missing)}.\n"
                f"Searched (in order):\n  - " + "\n  - ".join(searched)
            )
            raise RuntimeError(msg)

        url = os.environ[self.url_key]
        key = os.environ[self.service_key]

        if self.verbose:
            print(self.status_string())

        return url, key

    def status(self) -> Dict[str, str]:
        """
        Return a mapping of each required var -> where it was found.
        Values are one of: 'environment', '/abs/path/.env.secrets', '/abs/path/.env'
        """
        return dict(self._sources)

    def status_string(self) -> str:
        """
        Human-friendly status line(s) without exposing secret values.
        """
        parts = []
        for var in (self.url_key, self.service_key):
            src = self._sources.get(var, "<unresolved>")
            parts.append(f"✓ {var} found in: {src}")
        return "\n".join(parts)

    def create(self, factory: Callable[[str, str], object]) -> object:
        """
        Resolve creds and instantiate a client using the provided factory.
        Example:
            supa = ClientCreator("DB_API_URL", "DB_API_SERVICE_KEY").create(create_client)
        """
        url, key = self.resolve()
        return factory(url, key)

