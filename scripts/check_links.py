#!/usr/bin/env python3
"""Check every link in README.md (or files given) returns a non-error status. Stdlib only."""
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0 (link check)"}


def status(url):
    if url.startswith("https://doi.org/"):
        # publishers block scripts; Crossref confirms the DOI is registered
        url_api = "https://api.crossref.org/works/" + url[len("https://doi.org/"):]
        for attempt in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url_api, headers=UA), timeout=30) as r:
                    return url, r.status
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    return url, e.code
                time.sleep(2 ** attempt)        # Crossref rate limit
            except Exception as e:
                return url, type(e).__name__
        return url, 429
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, headers=UA, method=method)
            with urllib.request.urlopen(req, timeout=30) as r:
                return url, r.status
        except urllib.error.HTTPError as e:
            if method == "GET":
                return url, e.code
        except Exception as e:
            if method == "GET":
                return url, type(e).__name__
    return url, "?"


def main(paths):
    text = "\n".join(open(p).read() for p in paths)
    urls = sorted(set(re.findall(r"\]\((https?://[^)\s]+)\)", text))) or sorted(set(re.findall(r"https?://\S+", text)))
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(status, urls))
    bad = [(u, s) for u, s in res if s != 200]
    for u, s in bad:
        print(f"{s}\t{u}")
    print(f"{len(urls)} links, {len(bad)} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["README.md"]))
