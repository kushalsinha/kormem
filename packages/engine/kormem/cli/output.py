"""Output formats shared by search, query, get and multi-get: cli, json, csv, md, xml, files."""

from __future__ import annotations

import csv
import io
import json
import sys
from typing import Any, Dict, Iterable, List, Optional
from xml.sax.saxutils import escape, quoteattr

FORMATS = ("cli", "json", "csv", "md", "xml", "files")

_TTY = sys.stdout.isatty()


def dim(s: str) -> str:
    return f"\033[2m{s}\033[0m" if _TTY else s


def bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _TTY else s


def printJson(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def address(item: Dict[str, Any]) -> str:
    return item.get("uri") or item.get("docid") or item.get("title") or ""


def numbered(text: str, start: int = 1, enabled: bool = True) -> str:
    if not enabled:
        return text
    lines = text.split("\n")
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{str(start + i).rjust(width)}: {line}" for i, line in enumerate(lines))


def sliceLines(text: str, fromLine: int = 1, maxLines: Optional[int] = None) -> str:
    lines = text.split("\n")
    lo = max(fromLine, 1) - 1
    return "\n".join(lines[lo: lo + maxLines if maxLines else None])


def results(items: List[Dict[str, Any]], fmt: str, full: bool = False) -> None:
    """Ranked results: each item has docid, uri, title, score, context, text."""
    if fmt == "json":
        printJson(items)
    elif fmt == "files":
        for it in items:
            print(f"{address(it)}\t{it.get('score', 0):.3f}")
    elif fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["docid", "score", "uri", "title", "context", "text"])
        for it in items:
            w.writerow([it.get("docid", ""), f"{it.get('score', 0):.4f}", it.get("uri") or "",
                        it.get("title", ""), it.get("context", ""), it.get("text", "")])
        sys.stdout.write(buf.getvalue())
    elif fmt == "md":
        for it in items:
            print(f"## {it.get('title') or address(it)}\n")
            print(f"`{address(it)}` · {it.get('docid', '')} · score {it.get('score', 0):.3f}")
            if it.get("context"):
                print(f"\n> {it['context']}")
            print(f"\n{it.get('text', '')}\n")
    elif fmt == "xml":
        print("<results>")
        for it in items:
            print(f"  <result docid={quoteattr(it.get('docid', ''))} uri={quoteattr(it.get('uri') or '')} "
                  f"score=\"{it.get('score', 0):.4f}\">")
            print(f"    <title>{escape(it.get('title', ''))}</title>")
            if it.get("context"):
                print(f"    <context>{escape(it['context'])}</context>")
            print(f"    <text>{escape(it.get('text', ''))}</text>")
            print("  </result>")
        print("</results>")
    else:
        if not items:
            print("No results.")
        for it in items:
            print(f"{bold(address(it))}  {dim(it.get('docid', ''))}  {it.get('score', 0):.2f}")
            if it.get("title") and it.get("title") not in address(it):
                print(f"  {it['title']}")
            if it.get("context"):
                print(dim(f"  Context: {it['context']}"))
            text = it.get("text", "").strip()
            if not full:
                text = " ".join(text.split())[:280] + ("…" if len(text) > 280 else "")
            print("  " + text.replace("\n", "\n  ") + "\n")


def documents(items: Iterable[Dict[str, Any]], fmt: str, lineNumbers: bool = True) -> None:
    """Whole notes or sources from get / multi-get."""
    items = list(items)
    if fmt == "json":
        printJson(items if len(items) != 1 else items[0])
        return
    if fmt == "files":
        for it in items:
            print(address(it) if "error" not in it else f"{it['ref']}\t{it['error']}")
        return
    if fmt == "xml":
        print("<documents>")
        for it in items:
            if "error" in it:
                print(f"  <missing ref={quoteattr(it['ref'])}>{escape(it['error'])}</missing>")
                continue
            print(f"  <document uri={quoteattr(address(it))} docid={quoteattr(it.get('docid', ''))} "
                  f"date={quoteattr(it.get('date', ''))}>")
            print(f"    <title>{escape(it.get('title', ''))}</title>")
            print(f"    <text>{escape(it.get('text', ''))}</text>")
            print("  </document>")
        print("</documents>")
        return
    many = len(items) > 1
    for it in items:
        if "error" in it:
            print(f"{it['ref']}: {it['error']}", file=sys.stderr)
            continue
        if fmt == "md":
            print(f"## {it.get('title') or address(it)}\n\n`{address(it)}` {it.get('docid', '')} {it.get('date', '')}\n")
            print(it.get("text", "") + "\n")
            continue
        if many:
            print(bold(f"==> {address(it)} {it.get('docid', '')} <=="))
        if it.get("skipped"):
            print(dim(f"(skipped: {it['skipped']})\n"))
            continue
        print(numbered(it.get("text", ""), it.get("fromLine", 1), lineNumbers))
        if many:
            print()
