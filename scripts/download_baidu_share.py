#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def first_cookie_value(cookie_jar: requests.cookies.RequestsCookieJar, name: str) -> str:
    for cookie in cookie_jar:
        if cookie.name == name:
            return cookie.value
    return ""


def make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def extract_shorturl(share_url: str) -> str:
    parsed = urlparse(share_url)
    if parsed.path.startswith("/s/1"):
        return parsed.path.split("/s/1", 1)[1]
    match = re.search(r"surl=([^&]+)", share_url)
    if match:
        return match.group(1)
    raise ValueError(f"Unable to parse shorturl from share URL: {share_url}")


def verify_share(session: requests.Session, share_url: str, password: str) -> dict[str, Any]:
    shorturl = extract_shorturl(share_url)
    init = session.get(share_url, timeout=30)
    init.raise_for_status()
    verify = session.post(
        f"https://pan.baidu.com/share/verify?surl={shorturl}",
        data={"pwd": password, "vcode": "", "vcode_str": ""},
        headers={"Referer": share_url},
        timeout=30,
    )
    verify.raise_for_status()
    payload = verify.json()
    if payload.get("errno") != 0:
        raise RuntimeError(f"Share verification failed: {payload}")
    return payload


def list_share_files(session: requests.Session, shorturl: str, referer: str) -> dict[str, Any]:
    response = session.get(
        "https://pan.baidu.com/share/list",
        params={
            "app_id": "250528",
            "web": "1",
            "channel": "chunlei",
            "clienttype": "0",
            "page": "1",
            "num": "100",
            "order": "name",
            "desc": "1",
            "showempty": "0",
            "shorturl": shorturl,
            "root": "1",
        },
        headers={"Referer": referer},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errno") != 0:
        raise RuntimeError(f"Failed to list share files: {payload}")
    return payload


async def _browser_probe_download(share_page_url: str, cookies: dict[str, str]) -> dict[str, Any]:
    from pyppeteer import launch

    browser = await launch(
        headless=True,
        ignoreHTTPSErrors=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-zygote",
            "--single-process",
            "--no-proxy-server",
        ],
    )
    page = await browser.newPage()
    await page.setUserAgent(USER_AGENT)
    await page.setCookie(
        *[
            {"name": name, "value": value, "domain": ".pan.baidu.com", "path": "/"}
            for name, value in cookies.items()
            if value is not None
        ]
    )

    captured: dict[str, Any] = {}

    def on_request(req: Any) -> None:
        if "api/sharedownload" in req.url and "sharedownload_request_url" not in captured:
            captured["sharedownload_request_url"] = req.url
            captured["sharedownload_post_data"] = req.postData

    async def on_response(resp: Any) -> None:
        if "api/sharedownload" in resp.url and "sharedownload_response_text" not in captured:
            captured["sharedownload_response_text"] = await resp.text()

    page.on("request", on_request)
    page.on("response", lambda resp: asyncio.ensure_future(on_response(resp)))

    await page.goto(share_page_url, {"waitUntil": "domcontentloaded", "timeout": 120000})
    await asyncio.sleep(8)
    await page.waitForFunction("window.locals && window.locals.get('file_list')", {"timeout": 120000})
    await page.click("a.bottom_download_btn")
    await asyncio.sleep(8)

    payload = await page.evaluate(
        """() => ({
            url: location.href,
            shareid: window.locals ? window.locals.get('shareid') : null,
            share_uk: window.locals ? window.locals.get('share_uk') : null,
            file_list: window.locals ? window.locals.get('file_list') : null,
            currentSekey: window.currentSekey || null,
            title: document.title
        })"""
    )
    page_cookies = await page.cookies()
    await browser.close()
    payload["cookies"] = {item["name"]: item["value"] for item in page_cookies}
    payload.update(captured)
    return payload


def fetch_browser_probe(share_page_url: str, cookies: dict[str, str]) -> dict[str, Any]:
    return asyncio.run(_browser_probe_download(share_page_url, cookies))


def request_dlink(
    session: requests.Session,
    referer: str,
    sign: str,
    timestamp: str | int,
    share_id: int,
    share_uk: int,
    fs_id: int,
) -> dict[str, Any]:
    sekey = unquote(first_cookie_value(session.cookies, "BDCLND"))
    response = session.post(
        "https://pan.baidu.com/api/sharedownload",
        params={
            "sign": sign,
            "timestamp": str(timestamp),
            "bdstoken": "",
            "channel": "chunlei",
            "clienttype": "0",
            "web": "1",
            "app_id": "250528",
        },
        data={
            "encrypt": "0",
            "product": "share",
            "uk": str(share_uk),
            "primaryid": str(share_id),
            "fid_list": json.dumps([fs_id]),
            "extra": json.dumps({"sekey": sekey}, ensure_ascii=False),
        },
        headers={"Referer": referer},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errno") != 0:
        raise RuntimeError(f"Failed to obtain dlink: {payload}")
    return payload


def stream_download(
    session: requests.Session,
    download_url: str,
    output_path: Path,
    referer: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    existing = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Referer": referer}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    with session.get(download_url, headers=headers, stream=True, timeout=120) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(
                f"Download request failed with status {response.status_code}: {response.text[:300]}"
            )
        total = response.headers.get("Content-Length")
        total_bytes = int(total) + existing if total is not None else None
        mode = "ab" if existing else "wb"
        written = existing
        started = time.time()
        last_report = started
        with part_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_report >= 5:
                    if total_bytes:
                        pct = 100.0 * written / total_bytes
                        print(
                            f"[download] {written / (1024**3):.2f} / {total_bytes / (1024**3):.2f} GB "
                            f"({pct:.2f}%)"
                        )
                    else:
                        print(f"[download] {written / (1024**3):.2f} GB")
                    last_report = now
    part_path.rename(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share-url", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--metadata-name", default="baidu_share_metadata.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    share_url = args.share_url
    shorturl = extract_shorturl(share_url)
    output_dir = Path(args.output_dir).resolve()
    metadata_path = output_dir / args.metadata_name
    output_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    verify_payload = verify_share(session, share_url, args.password)
    list_payload = list_share_files(session, shorturl, share_url)
    files = list_payload.get("list", [])
    if len(files) != 1:
        raise RuntimeError(f"Expected one archive in share, got {len(files)} files")

    file_item = files[0]
    browser_payload = fetch_browser_probe(
        f"https://pan.baidu.com/s/1{shorturl}?from=init",
        session.cookies.get_dict(),
    )

    for name, value in browser_payload.get("cookies", {}).items():
        session.cookies.set(name, value, domain=".pan.baidu.com")
        session.cookies.set(name, value, domain="pan.baidu.com")

    sharedownload_url = browser_payload.get("sharedownload_request_url", "")
    sign_match = re.search(r"[?&]sign=([^&]+)", sharedownload_url)
    timestamp_match = re.search(r"[?&]timestamp=(\d+)", sharedownload_url)
    if not sign_match or not timestamp_match:
        raise RuntimeError("Failed to capture sharedownload sign/timestamp from browser probe.")

    sign = sign_match.group(1)
    timestamp = timestamp_match.group(1)
    share_id = int(browser_payload["shareid"])
    share_uk = int(browser_payload["share_uk"])
    fs_id = int(file_item["fs_id"])

    dlink_payload = request_dlink(
        session=session,
        referer=browser_payload["url"],
        sign=sign,
        timestamp=timestamp,
        share_id=share_id,
        share_uk=share_uk,
        fs_id=fs_id,
    )

    link_candidates = dlink_payload.get("list") or []
    dlink = None
    if isinstance(link_candidates, list) and link_candidates and isinstance(link_candidates[0], dict):
        dlink = link_candidates[0].get("dlink")
    filename = file_item["server_filename"]
    output_path = output_dir / filename

    metadata = {
        "share_url": share_url,
        "shorturl": shorturl,
        "verify": verify_payload,
        "share_listing": list_payload,
        "browser_payload": {
            "url": browser_payload["url"],
            "title": browser_payload.get("title"),
            "sign": sign,
            "timestamp": timestamp,
            "shareid": share_id,
            "share_uk": share_uk,
            "sharedownload_request_url": browser_payload.get("sharedownload_request_url"),
            "sharedownload_post_data": browser_payload.get("sharedownload_post_data"),
            "sharedownload_response_text": browser_payload.get("sharedownload_response_text"),
        },
        "download_payload": dlink_payload,
        "output_path": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if dlink:
        print(f"[info] downloading to {output_path}")
        stream_download(session, dlink, output_path, browser_payload["url"])
        print(f"[done] saved archive to {output_path}")
        return 0

    raise RuntimeError(
        "Baidu share returned an encrypted payload instead of a direct dlink. "
        "The script saved request metadata, but full anonymous download is currently blocked by the share flow."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
